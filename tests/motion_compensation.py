"""
motion_compensation.py — HEVC Inter-Picture Motion Compensation
Generates the inter prediction pixel block given a motion vector.

Pipeline position
-----------------
    motion_estimation.py  →  [motion_compensation.py]  →  transform/dct.py
    (MotionVector)             (pred pixels, residual)      (forward DCT)

Relationship to motion_estimation.py
--------------------------------------
    motion_estimation.py  — FINDS the best MV (search, cost evaluation)
    motion_compensation.py — EXECUTES the MV to generate prediction pixels

This file does three things ME does not:

    1. Uni-directional compensation  (P-frame, single reference)
       pred = interpolate(ref_L0, origin + MV)

    2. Bi-directional compensation   (B-frame, two references)
       pred = (interpolate(ref_L0, origin + MV_L0)
             + interpolate(ref_L1, origin + MV_L1) + 1) >> 1

    3. Weighted prediction           (HEVC §8.5.3.3)
       pred = clip((w * ref_pred + offset + (1 << (shift-1))) >> shift, 0, 255)
       Default weights w=1, offset=0 give standard averaging.

Interpolation
-------------
Luma uses the 8-tap filter from motion_estimation.py (imported directly).
Chroma uses a 4-tap filter at 1/8-pel precision (chroma block is half
the luma size in 4:2:0, so MV is halved and chroma uses its own filter).

HEVC chroma 4-tap filter (ISO/IEC 23008-2 §8.5.3.2.3):
    frac=0: [ 0, 64,  0,  0]   integer pel
    frac=1: [-2, 58,  10, -2]  1/8-pel
    frac=2: [-4, 54,  16, -2]  2/8 = 1/4-pel
    frac=3: [-6, 46,  28, -4]  3/8-pel
    frac=4: [-4, 36,  36, -4]  1/2-pel (symmetric)
    frac=5: [-4, 28,  46, -6]  5/8-pel
    frac=6: [-2, 16,  54, -4]  3/4-pel
    frac=7: [-2, 10,  58, -2]  7/8-pel

Coordinate convention (matches motion_estimation.py)
------------------------------------------------------
    MV is stored in QUARTER-PIXEL units.
    Integer MV (+3, -2) pixels → mvx=+12, mvy=-8 qpel units.

    For chroma (4:2:0): chroma MV = luma MV // 2  (in qpel units).
    Chroma precision is 1/8-pel (half luma qpel).

Public API
----------
    compensate_luma(ref_frame, origin, mv, n)                 -> np.ndarray
    compensate_chroma(ref_cb, ref_cr, origin_c, mv, n_c)      -> tuple
    compensate_bi(ref_l0, ref_l1, origin, mv_l0, mv_l1, n)   -> np.ndarray
    generate_inter_residual(block, pred)                       -> np.ndarray
    full_inter_pipeline(block, ref_frame, origin, mv, qp)     -> InterResult
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

# Reuse the luma interpolation engine from motion_estimation — single source of truth
from motion_estimation import (
    _interp_patch,
    _get_filter_coeffs,
    MotionVector,
)

# ---------------------------------------------------------------------------
# HEVC Chroma 4-tap interpolation filter (ISO/IEC 23008-2 §8.5.3.2.3)
# Indexed by fractional position in EIGHTH-pixel units (0..7).
# All filters sum to 64 — normalised by >>6 shift.
# ---------------------------------------------------------------------------
_CHROMA_FILTER: dict[int, list[int]] = {
    0: [ 0, 64,  0,  0],   # integer pel
    1: [-2, 58, 10, -2],   # 1/8-pel
    2: [-4, 54, 16, -2],   # 2/8 = 1/4-pel
    3: [-6, 46, 28, -4],   # 3/8-pel
    4: [-4, 36, 36, -4],   # 4/8 = 1/2-pel (symmetric)
    5: [-4, 28, 46, -6],   # 5/8-pel
    6: [-2, 16, 54, -4],   # 6/8 = 3/4-pel
    7: [-2, 10, 58, -2],   # 7/8-pel
}

# Weighted prediction defaults (HEVC §8.5.3.3 — default weights give equal blending)
_WP_DEFAULT_WEIGHT: int = 1
_WP_DEFAULT_OFFSET: int = 0
_WP_DEFAULT_SHIFT:  int = 0   # no extra shift for default case


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class InterResult:
    """Complete output of the inter prediction pipeline for one luma block."""
    mv:           MotionVector   # motion vector used
    pred:         np.ndarray     # prediction pixels,  (N, N), uint8
    residual:     np.ndarray     # original − pred,    (N, N), int16
    dct_coeffs:   np.ndarray | None = None   # DCT coefficients, int32
    quant_levels: np.ndarray | None = None   # quantised levels, int32


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compensate_luma(
    ref_frame: np.ndarray,
    origin:    tuple[int, int],
    mv:        MotionVector | tuple[int, int],
    n:         int,
) -> np.ndarray:
    """
    Generate a luma prediction block using a motion vector.

    Applies the HEVC 8-tap luma interpolation filter at fractional positions.
    Integer-pel MVs produce an exact pixel copy from the reference.

    Parameters
    ----------
    ref_frame : np.ndarray
        Full reference luma frame, shape (H, W), dtype uint8.
    origin : (ox, oy)
        Top-left corner of the current block in the reference frame
        (integer pixel coordinates). x = column, y = row.
    mv : MotionVector or (mvx, mvy)
        Motion vector in quarter-pixel units.
        A plain tuple (mvx, mvy) is also accepted for convenience.
    n : int
        Block size (luma samples). Must be in {4, 8, 16, 32, 64}.

    Returns
    -------
    pred : np.ndarray
        Interpolated prediction block, shape (n, n), dtype uint8,
        values clipped to [0, 255].

    Raises
    ------
    ValueError
        If the interpolated patch falls outside the reference frame bounds.
    """
    ox, oy = origin
    mvx, mvy = (mv.mvx, mv.mvy) if isinstance(mv, MotionVector) else mv

    # Absolute position in qpel units
    abs_qx = ox * 4 + mvx
    abs_qy = oy * 4 + mvy

    _validate_patch_bounds(ref_frame, abs_qx, abs_qy, n, "luma")

    # Integer and fractional parts
    ix = abs_qx // 4
    iy = abs_qy // 4
    fx = abs_qx % 4   # 0..3 quarter-pel fraction
    fy = abs_qy % 4

    # Convert qpel fractions to eighth-pel for the filter table (× 2)
    return _interp_patch(ref_frame, ix, iy, fx * 2, fy * 2, n)


def compensate_chroma(
    ref_cb:   np.ndarray,
    ref_cr:   np.ndarray,
    origin_c: tuple[int, int],
    mv:       MotionVector | tuple[int, int],
    n_c:      int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Cb and Cr chroma prediction blocks using a luma motion vector.

    In 4:2:0, the chroma block is half the luma size. The luma MV (in qpel
    units) is divided by 2 to get the chroma MV (in chroma 1/8-pel units).
    The HEVC 4-tap chroma interpolation filter is used.

    Parameters
    ----------
    ref_cb : np.ndarray
        Full Cb reference plane, shape (H/2, W/2), dtype uint8.
    ref_cr : np.ndarray
        Full Cr reference plane, shape (H/2, W/2), dtype uint8.
    origin_c : (ox_c, oy_c)
        Top-left of the chroma block in the chroma reference plane
        (integer chroma-pixel coordinates).
    mv : MotionVector or (mvx, mvy)
        Luma MV in quarter-pixel units. Divided by 2 internally.
    n_c : int
        Chroma block size. Typically n_luma // 2 for 4:2:0.

    Returns
    -------
    (pred_cb, pred_cr) : tuple of np.ndarray
        Both shape (n_c, n_c), dtype uint8.
    """
    mvx, mvy = (mv.mvx, mv.mvy) if isinstance(mv, MotionVector) else mv

    # Chroma MV unit conversion (4:2:0):
    #   luma qpel      = luma pixel × 4
    #   chroma pixel   = luma pixel / 2     (4:2:0 horizontal/vertical subsampling)
    #   chroma 8th-pel = chroma pixel × 8
    #
    #   chroma 8th-pel = (luma qpel / 4) / 2 × 8 = luma qpel × 1
    #
    # Therefore: cmv_eighth_pel == luma_mv_qpel  (same numerical value)
    cmvx = mvx   # chroma MV in eighth-pel units
    cmvy = mvy

    ox_c, oy_c = origin_c
    abs_cx = ox_c * 8 + cmvx   # absolute chroma position in eighth-pel
    abs_cy = oy_c * 8 + cmvy

    ix = abs_cx // 8
    iy = abs_cy // 8
    fx = abs_cx % 8   # 0..7 eighth-pel fraction
    fy = abs_cy % 8

    pred_cb = _interp_chroma(ref_cb, ix, iy, fx, fy, n_c)
    pred_cr = _interp_chroma(ref_cr, ix, iy, fx, fy, n_c)

    return pred_cb, pred_cr


def compensate_bi(
    ref_l0:  np.ndarray,
    ref_l1:  np.ndarray,
    origin:  tuple[int, int],
    mv_l0:   MotionVector | tuple[int, int],
    mv_l1:   MotionVector | tuple[int, int],
    n:       int,
    weight_l0: int = 1,
    weight_l1: int = 1,
    offset:    int = 0,
) -> np.ndarray:
    """
    Bi-directional motion compensation (B-slice prediction).

    Generates a weighted average of two reference predictions:

        pred = clip(( w0 * pred_L0 + w1 * pred_L1 + offset*2 + 1 ) >> 1, 0, 255)

    Default (w0=w1=1, offset=0) gives equal-weight averaging — the most
    common case in HEVC B-slices.

    Parameters
    ----------
    ref_l0, ref_l1 : np.ndarray
        Reference luma frames for list 0 and list 1, shape (H, W), uint8.
    origin : (ox, oy)
        Block origin in integer pixel coordinates (same for both refs).
    mv_l0, mv_l1 : MotionVector or (mvx, mvy)
        Motion vectors in quarter-pixel units for each reference list.
    n : int
        Block size.
    weight_l0, weight_l1 : int
        Weighted prediction multipliers (default 1 each — equal weight).
    offset : int
        Weighted prediction additive offset (default 0).

    Returns
    -------
    pred : np.ndarray
        Bi-predicted block, shape (n, n), dtype uint8, clipped [0, 255].
    """
    pred_0 = compensate_luma(ref_l0, origin, mv_l0, n).astype(np.int32)
    pred_1 = compensate_luma(ref_l1, origin, mv_l1, n).astype(np.int32)

    # Weighted average: (w0*p0 + w1*p1 + offset*2 + 1) >> 1
    blended = (weight_l0 * pred_0 + weight_l1 * pred_1 + offset * 2 + 1) >> 1
    return np.clip(blended, 0, 255).astype(np.uint8)


def generate_inter_residual(
    original: np.ndarray,
    pred:     np.ndarray,
) -> np.ndarray:
    """
    Compute the inter residual: original − prediction.

    Identical contract to intra_prediction.generate_residual().
    The result feeds into forward_dct() in transform/dct.py.

    Parameters
    ----------
    original : (N, N) uint8 or int16 — original luma block
    pred     : (N, N) uint8          — compensated prediction

    Returns
    -------
    residual : (N, N) int16
        Signed difference, values in approximately [−255, 255].

    Raises
    ------
    ValueError — shape mismatch.
    """
    if original.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs pred {pred.shape}."
        )
    return (original.astype(np.int16) - pred.astype(np.int16))


def full_inter_pipeline(
    block:     np.ndarray,
    ref_frame: np.ndarray,
    origin:    tuple[int, int],
    mv:        MotionVector,
    qp:        int = 28,
) -> InterResult:
    """
    Run the complete inter encoding pipeline for one luma block.

    Pipeline:
        compensate_luma → generate_inter_residual → forward_dct → quantize

    Parameters
    ----------
    block     : (N, N) uint8    — original luma pixels
    ref_frame : (H, W) uint8   — reference luma frame
    origin    : (ox, oy)        — block origin in integer pixels
    mv        : MotionVector    — motion vector in qpel units
    qp        : int             — quantisation parameter [0, 51]

    Returns
    -------
    InterResult with pred, residual, dct_coeffs, quant_levels populated.
    """
    from dct import forward_dct
    from quantizer import quantize

    n    = block.shape[0]
    pred = compensate_luma(ref_frame, origin, mv, n)
    res  = generate_inter_residual(block, pred)
    coeffs = forward_dct(res)
    levels = quantize(coeffs, qp=qp, is_intra=False)  # inter dead-zone

    return InterResult(
        mv=mv,
        pred=pred,
        residual=res,
        dct_coeffs=coeffs,
        quant_levels=levels,
    )


# ---------------------------------------------------------------------------
# Chroma interpolation kernel (4-tap)
# ---------------------------------------------------------------------------

def _interp_chroma(
    ref: np.ndarray,
    ix:  int,
    iy:  int,
    fx:  int,   # fractional x in eighth-pel units (0..7)
    fy:  int,   # fractional y in eighth-pel units (0..7)
    n:   int,
) -> np.ndarray:
    """
    2-D separable 4-tap chroma interpolation.

    Two-pass: horizontal first (filter each row), then vertical.
    Rounding shift >>6 after each pass (same as luma).
    """
    H, W = ref.shape
    tap    = 4
    extra  = tap - 1          # 3 extra samples needed beyond block
    OFFSET = 1                # chroma filter identity tap is at index 1

    h_coeffs = _CHROMA_FILTER[fx]
    v_coeffs = _CHROMA_FILTER[fy]

    rows_needed = n + extra

    def get_row(y_idx: int, x_start: int) -> np.ndarray:
        y_idx = max(0, min(y_idx, H - 1))
        xs    = np.arange(x_start, x_start + n + extra)
        xs    = np.clip(xs, 0, W - 1)
        return ref[y_idx, xs].astype(np.int32)

    # Horizontal pass: start at ix - OFFSET to align identity tap
    h_rows = np.zeros((rows_needed, n), dtype=np.int32)
    for i in range(rows_needed):
        row = get_row(iy - OFFSET + i, ix - OFFSET)
        for col in range(n):
            acc = sum(h_coeffs[t] * int(row[col + t]) for t in range(tap))
            h_rows[i, col] = acc

    h_rows = (h_rows + 32) >> 6   # round

    # Vertical pass
    patch = np.zeros((n, n), dtype=np.int32)
    for col in range(n):
        for row in range(n):
            acc = sum(v_coeffs[t] * int(h_rows[row + t, col]) for t in range(tap))
            patch[row, col] = (acc + 32) >> 6

    return np.clip(patch, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_patch_bounds(
    ref:   np.ndarray,
    abs_qx: int,
    abs_qy: int,
    n:     int,
    plane: str,
) -> None:
    """Ensure the interpolated patch lies within the reference frame."""
    H, W = ref.shape
    ix = abs_qx // 4
    iy = abs_qy // 4
    # The 8-tap filter reads from ix-3 to ix+n+4
    if ix - 3 < 0 or iy - 3 < 0 or ix + n + 4 > W or iy + n + 4 > H:
        raise ValueError(
            f"MV places {plane} patch out of reference frame bounds. "
            f"abs_q=({abs_qx},{abs_qy}), n={n}, frame={W}×{H}."
        )