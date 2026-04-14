"""
intra_prediction.py — HEVC Intra-Picture Prediction (Pixel Generation)
Generates the actual prediction pixel block given a chosen mode.

Pipeline position
-----------------
    intra_estimation.py  →  [intra_prediction.py]  →  transform/dct.py
    (best mode index)        (pred pixels, residual)   (forward DCT)

Relationship to intra_estimation.py
------------------------------------
    intra_estimation.py  — DECIDES which mode wins (SAD/SATD cost search)
    intra_prediction.py  — EXECUTES the winning mode to generate pred pixels

This file adds three things not in intra_estimation.py:

    1. Reference sample filtering   (§8.4.4.2.3)
       Before any prediction kernel runs, the reference samples may be
       smoothed to reduce blocking artefacts, especially for large blocks.

    2. DC boundary post-filter      (§8.4.4.2.5)
       After DC prediction, a [1,2,1]/4 filter corrects the first row
       and first column toward the reference samples at the block edges.

    3. Chroma intra prediction      (§8.4.5)
       HEVC defines 5 chroma modes for Cb and Cr. Chroma uses a smaller
       block (half luma size for 4:2:0) and includes DM mode (Derived
       Mode), which mirrors the co-located luma mode choice.

Reference sample layout (HEVC convention)
------------------------------------------
    ref_above: 1-D array, length 2N+1
        [corner, top[0], top[1], ..., top[2N-1]]
        ref_above[0]    = top-left corner pixel
        ref_above[1:N+1] = N samples directly above the block

    ref_left: 1-D array, length 2N+1
        [corner, left[0], left[1], ..., left[2N-1]]
        ref_left[0]     = same top-left corner as ref_above[0]
        ref_left[1:N+1] = N samples directly to the left

Public API
----------
    predict_intra_luma(mode, ref_above, ref_left, n,
                       filter_refs, apply_dc_filter) -> np.ndarray
    predict_intra_chroma(chroma_mode, ref_above_c, ref_left_c, n_c,
                         luma_mode)                  -> np.ndarray
    generate_residual(original, pred)                -> np.ndarray
    full_intra_pipeline(block, mode, ref_above, ref_left,
                        qp, is_intra)                -> IntraResult
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

# Re-use prediction kernels from intra_estimation — single source of truth
from prediction.intra_estimation import (
    _pred_planar,
    _pred_dc,
    _pred_angular,
    PLANAR_MODE, DC_MODE,
    VALID_BLOCK_SIZES,
)

# ---------------------------------------------------------------------------
# Chroma mode constants (HEVC §8.4.5)
# ---------------------------------------------------------------------------

# 5 chroma intra modes
CHROMA_PLANAR:     int = 0   # same as luma planar
CHROMA_VERTICAL:   int = 1   # mode 26 from luma table
CHROMA_HORIZONTAL: int = 2   # mode 10 from luma table
CHROMA_DC:         int = 3   # same as luma DC
CHROMA_DM:         int = 4   # Derived Mode — mirror the luma mode

# Chroma mode → luma mode equivalent (for prediction kernel dispatch)
_CHROMA_TO_LUMA_MODE: dict[int, int] = {
    CHROMA_PLANAR:     PLANAR_MODE,   # 0
    CHROMA_VERTICAL:   26,            # pure vertical
    CHROMA_HORIZONTAL: 10,            # pure horizontal
    CHROMA_DC:         DC_MODE,       # 1
    # CHROMA_DM is resolved at runtime from the luma mode argument
}

# Reference filter threshold for strong intra smoothing (spec §8.4.4.2.3)
# If max - min of corner ref samples exceeds this, skip strong smoothing.
_STRONG_SMOOTH_THRESHOLD: int = 8


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class IntraResult:
    """Complete output of the intra prediction pipeline for one block."""
    mode:        int            # luma intra mode used [0, 34]
    pred:        np.ndarray     # prediction pixels,  shape (N, N), uint8
    residual:    np.ndarray     # original - pred,    shape (N, N), int16
    # Populated by full_intra_pipeline when transform is included
    dct_coeffs:  np.ndarray | None = None   # DCT coefficients, int32
    quant_levels: np.ndarray | None = None  # quantised levels, int32


# ---------------------------------------------------------------------------
# Public API — Luma
# ---------------------------------------------------------------------------

def predict_intra_luma(
    mode:             int,
    ref_above:        np.ndarray,
    ref_left:         np.ndarray,
    n:                Literal[4, 8, 16, 32] | None = None,
    filter_refs:      bool = True,
    apply_dc_filter:  bool = True,
) -> np.ndarray:
    """
    Generate the luma intra prediction block for a given mode.

    This is the pixel-generation counterpart to intra_estimation.py.
    The caller has already decided the best mode; this function executes it.

    Processing order (per spec §8.4.4):
        1. [Optional] Filter reference samples          (§8.4.4.2.3)
        2. Run the prediction kernel for the mode       (§8.4.4.2.4–6)
        3. [Optional] DC boundary post-filter           (§8.4.4.2.5)

    Parameters
    ----------
    mode : int
        Intra prediction mode index [0, 34].
    ref_above : np.ndarray
        Reference samples above the block, shape (2N+1,), int16.
    ref_left : np.ndarray
        Reference samples to the left, shape (2N+1,), int16.
    n : {4, 8, 16, 32} or None
        Block size. Inferred from ref_above length if None.
    filter_refs : bool
        Apply reference sample smoothing before prediction (default True).
        Disable for debugging or when refs are pre-filtered.
    apply_dc_filter : bool
        Apply DC boundary post-filter for mode 1 (default True).

    Returns
    -------
    pred : np.ndarray
        Prediction block, shape (N, N), dtype uint8, values in [0, 255].

    Raises
    ------
    ValueError
        Invalid mode, mismatched array sizes, or unsupported block size.
    """
    n = _infer_and_validate_n(n, ref_above, ref_left)
    _validate_mode(mode)

    ref_a = ref_above.astype(np.int32)
    ref_l = ref_left.astype(np.int32)

    # Step 1 — Reference sample filtering
    if filter_refs:
        ref_a, ref_l = _filter_reference_samples(ref_a, ref_l, n, mode)

    # Step 2 — Prediction kernel
    if mode == PLANAR_MODE:
        pred = _pred_planar(ref_a, ref_l, n)
    elif mode == DC_MODE:
        pred = _pred_dc(ref_a, ref_l, n)
        # Step 3 — DC boundary post-filter
        if apply_dc_filter and n < 32:
            pred = _dc_post_filter(pred, ref_a, ref_l, n)
    else:
        pred = _pred_angular(mode, ref_a, ref_l, n)

    return np.clip(pred, 0, 255).astype(np.uint8)


def predict_intra_chroma(
    chroma_mode:   int,
    ref_above_c:   np.ndarray,
    ref_left_c:    np.ndarray,
    n_c:           Literal[2, 4, 8, 16] | None = None,
    luma_mode:     int | None = None,
    filter_refs:   bool = True,
) -> np.ndarray:
    """
    Generate a chroma (Cb or Cr) intra prediction block.

    HEVC §8.4.5 defines 5 chroma intra modes:
        0 — PLANAR     : same bilinear blend as luma planar
        1 — VERTICAL   : copy top reference down (≡ luma mode 26)
        2 — HORIZONTAL : copy left reference right (≡ luma mode 10)
        3 — DC         : mean of top and left reference samples
        4 — DM         : use the same mode as the co-located luma block

    For 4:2:0, the chroma block is half the luma size in each dimension.
    This function accepts any power-of-2 block size in [2, 16].

    Parameters
    ----------
    chroma_mode : int
        Chroma intra mode [0, 4]. Use CHROMA_* constants.
    ref_above_c : np.ndarray
        Chroma reference samples above, shape (2*N_c+1,), int16.
    ref_left_c : np.ndarray
        Chroma reference samples left, shape (2*N_c+1,), int16.
    n_c : {2, 4, 8, 16} or None
        Chroma block size. Inferred from ref_above_c length if None.
    luma_mode : int or None
        The luma intra mode for the co-located CU. Required when
        chroma_mode == CHROMA_DM (4). Ignored otherwise.
    filter_refs : bool
        Apply reference sample smoothing (default True).

    Returns
    -------
    pred : np.ndarray
        Chroma prediction block, shape (N_c, N_c), dtype uint8.

    Raises
    ------
    ValueError
        Invalid chroma_mode, or DM mode with no luma_mode provided.
    """
    if chroma_mode not in range(5):
        raise ValueError(f"Invalid chroma_mode {chroma_mode}. Must be 0–4.")
    if chroma_mode == CHROMA_DM and luma_mode is None:
        raise ValueError("luma_mode must be provided when chroma_mode == CHROMA_DM.")

    # Resolve DM → actual luma mode
    effective_luma_mode = (
        luma_mode if chroma_mode == CHROMA_DM
        else _CHROMA_TO_LUMA_MODE[chroma_mode]
    )

    # Chroma block sizes are not restricted to VALID_BLOCK_SIZES (can be 2)
    n_c_val = _infer_n_from_ref(ref_above_c) if n_c is None else n_c

    ref_a = ref_above_c.astype(np.int32)
    ref_l = ref_left_c.astype(np.int32)

    # Chroma reference filtering (lighter than luma — no strong smoothing)
    if filter_refs and n_c_val >= 4:
        ref_a, ref_l = _filter_reference_samples(
            ref_a, ref_l, n_c_val, effective_luma_mode, strong_smooth=False
        )

    # Dispatch to the same luma prediction kernels
    if effective_luma_mode == PLANAR_MODE:
        pred = _pred_planar(ref_a, ref_l, n_c_val)
    elif effective_luma_mode == DC_MODE:
        pred = _pred_dc(ref_a, ref_l, n_c_val)
    else:
        pred = _pred_angular(effective_luma_mode, ref_a, ref_l, n_c_val)

    return np.clip(pred, 0, 255).astype(np.uint8)


def generate_residual(
    original: np.ndarray,
    pred:     np.ndarray,
) -> np.ndarray:
    """
    Compute the residual block: original − prediction.

    This is the input to the forward DCT in transform/dct.py.

    Parameters
    ----------
    original : np.ndarray — original pixel block (uint8 or int16), shape (N, N)
    pred     : np.ndarray — prediction block (uint8), same shape

    Returns
    -------
    residual : np.ndarray
        Signed difference, dtype int16, shape (N, N).
        Values in approximately [-255, 255] for 8-bit input.
    """
    if original.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: original {original.shape} vs pred {pred.shape}."
        )
    return (original.astype(np.int16) - pred.astype(np.int16))


def full_intra_pipeline(
    block:     np.ndarray,
    mode:      int,
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
    qp:        int = 28,
    is_intra:  bool = True,
) -> IntraResult:
    """
    Run the complete luma intra encoding pipeline for one block.

    Pipeline:
        predict_intra_luma → generate_residual → forward_dct → quantize

    Parameters
    ----------
    block     : (N, N) uint8 — original luma pixels
    mode      : int          — intra prediction mode [0, 34]
    ref_above : (2N+1,) int16
    ref_left  : (2N+1,) int16
    qp        : int          — quantisation parameter [0, 51]
    is_intra  : bool         — intra dead-zone (True) or inter (False)

    Returns
    -------
    IntraResult with pred, residual, dct_coeffs, quant_levels populated.
    """
    from transform.dct import forward_dct
    from transform.quantizer import quantize

    pred     = predict_intra_luma(mode, ref_above, ref_left)
    residual = generate_residual(block, pred)
    coeffs   = forward_dct(residual)
    levels   = quantize(coeffs, qp=qp, is_intra=is_intra)

    return IntraResult(
        mode=mode,
        pred=pred,
        residual=residual,
        dct_coeffs=coeffs,
        quant_levels=levels,
    )


# ---------------------------------------------------------------------------
# Reference sample filtering (ISO/IEC 23008-2 §8.4.4.2.3)
# ---------------------------------------------------------------------------

def _filter_reference_samples(
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
    n:         int,
    mode:      int,
    strong_smooth: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Filter reference samples before prediction.

    Two filter modes are defined by the spec:

    Strong intra smoothing (32×32 blocks only, §8.4.4.2.3 condition):
        Applied when the top and left reference corner-to-corner variation is
        small (≤ threshold). Uses bilinear interpolation to create very smooth
        reference arrays. Intended to suppress blocking artefacts in large,
        smooth blocks.

    Weak [1, 2, 1] / 4 filter (all other eligible blocks):
        A 3-tap smoothing filter. Applied when the mode is not strongly
        directional (i.e., not a near-vertical or near-horizontal angular mode
        that would alias with the filter).

    Filtering is skipped for:
        - DC mode (mode 1): DC prediction averages refs itself
        - 4×4 blocks: too small, filter would damage angular accuracy
        - Strongly directional modes with small block sizes

    Returns filtered copies of ref_above and ref_left.
    """
    # 4×4 blocks are never filtered (spec §8.4.4.2.3 threshold table)
    if n == 4:
        return ref_above.copy(), ref_left.copy()

    # DC mode: no filtering (DC kernel averages internally)
    if mode == DC_MODE:
        return ref_above.copy(), ref_left.copy()

    # Determine filter eligibility based on mode "distance from cardinal axes"
    # Spec defines a threshold filterFlag per (mode, blockSize) combination.
    # Simplified: filter when mode is non-cardinal (not 10 or 26 exactly)
    # and intra smoothing is not disabled by a strongly diagonal mode.
    if not _should_filter(mode, n):
        return ref_above.copy(), ref_left.copy()

    # Strong intra smoothing for 32×32 (§8.4.4.2.3)
    if strong_smooth and n == 32:
        bi_above = int(ref_above[0])     # top-left corner
        bi_above_r = int(ref_above[2*n])  # top-right corner
        bi_left_b  = int(ref_left[2*n])   # bottom-left corner

        above_smooth = (abs(bi_above + bi_above_r - 2 * int(ref_above[n]))
                        <= _STRONG_SMOOTH_THRESHOLD)
        left_smooth  = (abs(bi_above + bi_left_b  - 2 * int(ref_left[n]))
                        <= _STRONG_SMOOTH_THRESHOLD)

        fa = ref_above.copy()
        fl = ref_left.copy()

        if above_smooth:
            # Bilinear interpolation of top reference
            for i in range(1, 2*n):
                fa[i] = ((2*n - i) * bi_above + i * bi_above_r + n) >> (int(np.log2(n)) + 1)

        if left_smooth:
            # Bilinear interpolation of left reference
            for i in range(1, 2*n):
                fl[i] = ((2*n - i) * bi_above + i * bi_left_b + n) >> (int(np.log2(n)) + 1)

        return fa, fl

    # Weak [1, 2, 1] / 4 smoothing filter
    fa = ref_above.copy()
    fl = ref_left.copy()

    # Filter top reference: fa[i] = (ref[i-1] + 2*ref[i] + ref[i+1] + 2) >> 2
    # Keep corners unchanged (i=0 and i=2N)
    ref_a_orig = ref_above.copy()
    ref_l_orig = ref_left.copy()

    for i in range(1, 2 * n):
        fa[i] = (ref_a_orig[i-1] + 2 * ref_a_orig[i] + ref_a_orig[i+1] + 2) >> 2 \
                if i < 2 * n else ref_a_orig[i]

    for i in range(1, 2 * n):
        fl[i] = (ref_l_orig[i-1] + 2 * ref_l_orig[i] + ref_l_orig[i+1] + 2) >> 2 \
                if i < 2 * n else ref_l_orig[i]

    return fa, fl


def _should_filter(mode: int, n: int) -> bool:
    """
    Return True if reference sample filtering should be applied.

    Per HEVC spec §8.4.4.2.3 Table 8-3 (simplified):
    Filtering is disabled for modes very close to cardinal directions
    (within ±1 mode step) to prevent the filter from blurring sharp edges.

    The threshold varies by block size:
        N= 4: no filtering at all (handled above)
        N= 8: filter disabled within 7 modes of vertical or horizontal
        N=16: filter disabled within 1 mode of vertical or horizontal
        N=32: strong smoothing, see _filter_reference_samples
    """
    if mode == PLANAR_MODE:
        return True  # always filter for planar

    # Angular modes: check distance from cardinal axes (modes 10 and 26)
    dist_vert = abs(mode - 26)   # distance from pure vertical
    dist_horiz = abs(mode - 10)  # distance from pure horizontal

    thresholds: dict[int, int] = {8: 7, 16: 1, 32: 0}
    thresh = thresholds.get(n, 0)

    # Filter if mode is not too close to a cardinal direction
    return dist_vert > thresh and dist_horiz > thresh


# ---------------------------------------------------------------------------
# DC boundary post-filter (ISO/IEC 23008-2 §8.4.4.2.5)
# ---------------------------------------------------------------------------

def _dc_post_filter(
    pred:      np.ndarray,
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
    n:         int,
) -> np.ndarray:
    """
    Apply the DC boundary post-filter for blocks smaller than 32×32.

    After DC prediction (which fills every sample with the same dc_val),
    the first row and first column are adjusted toward the reference samples
    at the block boundary using a [1, 2, 1]/4 blend:

        pred[0, 0]    = (ref_above[0] + 2*dc + ref_left[0+1] + 2) >> 2
                        using the corner for x=0, y=0 specifically:
                        (ref_left[1] + 2*dc + ref_above[1] + 2) >> 2

        pred[0, x]    = (ref_above[x+1] + 3*dc + 2) >> 2   for x > 0
        pred[y, 0]    = (ref_left[y+1]  + 3*dc + 2) >> 2   for y > 0

    This correction reduces the visible seam between the predicted block
    and its reconstructed neighbours.
    """
    out    = pred.copy()
    dc_val = int(pred[0, 0])   # all samples are dc_val at this point

    # Top-left corner: blend with both top and left reference
    out[0, 0] = np.clip(
        (int(ref_above[1]) + 2 * dc_val + int(ref_left[1]) + 2) >> 2,
        0, 255
    )
    # First row (x > 0): blend with top reference
    for x in range(1, n):
        out[0, x] = np.clip((int(ref_above[x+1]) + 3 * dc_val + 2) >> 2, 0, 255)
    # First column (y > 0): blend with left reference
    for y in range(1, n):
        out[y, 0] = np.clip((int(ref_left[y+1]) + 3 * dc_val + 2) >> 2, 0, 255)

    return out.astype(np.int16)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_and_validate_n(
    n:         int | None,
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
) -> int:
    """Infer block size from reference array length and validate."""
    n_inferred = _infer_n_from_ref(ref_above)
    n_val = n if n is not None else n_inferred

    if n_val not in VALID_BLOCK_SIZES:
        raise ValueError(
            f"Unsupported block size {n_val}. Supported: {VALID_BLOCK_SIZES}."
        )
    exp = 2 * n_val + 1
    if len(ref_above) != exp:
        raise ValueError(
            f"ref_above must have length {exp} for N={n_val}, got {len(ref_above)}."
        )
    if len(ref_left) != exp:
        raise ValueError(
            f"ref_left must have length {exp} for N={n_val}, got {len(ref_left)}."
        )
    return n_val


def _infer_n_from_ref(ref: np.ndarray) -> int:
    """Derive block size N from reference array length: len = 2N+1 → N."""
    return (len(ref) - 1) // 2


def _validate_mode(mode: int) -> None:
    if not (0 <= mode <= 34):
        raise ValueError(f"Invalid intra mode {mode}. Must be in [0, 34].")