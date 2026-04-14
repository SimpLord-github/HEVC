"""
motion_estimation.py — HEVC Inter-Picture Motion Estimation
Finds the best motion vector (MV) for a given luma block in a reference frame.

Pipeline position
-----------------
    decoded_picture_buffer.py  →  [motion_estimation.py]  →  motion_compensation.py
    (reference frames)             (motion vector)             (generate pred pixels)

Theory: what motion estimation does
-------------------------------------
Inter prediction exploits temporal redundancy between frames. For each block
in the current frame, ME finds the position in a reference frame that best
matches it. The offset between the two positions is the Motion Vector (MV).

    current_block ≈ ref_frame[y + mv_y, x + mv_x, ...]

The encoder only transmits the residual (difference) and the MV — not the
full block. Good MVs → small residuals → fewer bits after DCT+Q.

HEVC search range and precision
---------------------------------
HEVC supports quarter-pixel (1/4-pel) MV precision for luma.
The search proceeds in three stages:

    Stage 1 — Integer-pel search:
        Search on full-pixel grid within ±search_range pixels.
        Three algorithms available:
            FULL  (exhaustive) — test every integer candidate
            HEX   (hexagonal)  — x265-style 6-point hex pattern with refinement
            UMH   (Uneven Multi-Hexagon) — large-step global then fine local

    Stage 2 — Half-pixel refinement:
        8-point diamond search around the integer-pel winner.
        Interpolate reference frame to half-pixel grid using 6-tap HEVC filter.

    Stage 3 — Quarter-pixel refinement:
        8-point diamond search at 1/4-pel precision around the half-pel winner.
        Interpolate using the same 6-tap filter applied fractionally.

Coordinate convention
-----------------------
    MV is stored as (mv_x, mv_y) in QUARTER-PIXEL units.
    Integer MV of (+2, -3) pixels is stored as (+8, -12) in qpel units.
    This matches the HEVC spec (ISO/IEC 23008-2 §8.5.3) and hardware convention.

Cost function
-------------
    SAD (Sum of Absolute Differences) — used for speed at integer-pel stage.
    SATD (Sum of Absolute Hadamard Differences) — used for half/qpel refinement.

Public API
----------
    estimate_motion(block, ref_frame, origin, search_range,
                    algorithm, use_satd_refine)  -> MotionVector
    compute_sad(block, ref_patch)               -> int
    compute_satd(block, ref_patch)              -> int
    interpolate_half_pel(ref_frame, x, y, n)   -> np.ndarray
    interpolate_qpel(ref_frame, qx, qy, n)     -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_BLOCK_SIZES = (4, 8, 16, 32, 64)   # luma CU sizes + CTU

# Search algorithms
ALGO_FULL = "full"   # exhaustive — accurate but slow, O(range²)
ALGO_HEX  = "hex"    # hexagonal — x265 default, ~O(log range)
ALGO_UMH  = "umh"    # uneven multi-hexagon — balance of global/local

# Default search range (integer pixels). x265 default is 57.
DEFAULT_SEARCH_RANGE: int = 32

# HEVC 8-tap luma interpolation filter coefficients (ISO/IEC 23008-2 Table 8-10)
# Indexed by fractional position in QUARTER-PIXEL units (0=integer, 1=qpel, 2=hpel, 3=3qpel).
# The filter for fraction f and (4-f) are reverses of each other (symmetry property).
# All filters sum to 64 — normalised by the >>6 shift in _interp_patch.
#
# Internal use: we convert qpel fractions (0-3) to eighth-pixel units (0,2,4,6)
# via frac_eighth = frac_qpel * 2, then _get_filter_coeffs handles the lookup.
_HEVC_LUMA_FILTER: dict[int, list[int]] = {
    0: [ 0,  0,   0, 64,  0,   0,  0,  0],  # integer pel (copy)
    2: [-1,  4, -10, 58, 17,  -5,  1,  0],  # 1/4-pel
    4: [-1,  4, -11, 40, 40, -11,  4, -1],  # 1/2-pel (symmetric)
    6: [ 0,  1,  -5, 17, 58, -10,  4, -1],  # 3/4-pel  (= reverse of 1/4-pel)
    # Entries for intermediate eighth-pel positions (used when frac_eighth is odd):
    1: [-1,  4, -10, 58, 17,  -5,  1,  0],  # 1/8-pel  (≈ 1/4-pel)
    3: [ 0,  1,  -5, 17, 58, -10,  4, -1],  # 3/8-pel  (≈ 3/4-pel)
}

# Hexagonal search pattern (6 neighbours + centre)
_HEX_PATTERN = [
    (-2, 0), (-1, -2), (1, -2), (2, 0), (1, 2), (-1, 2),
]

# Large hexagon for UMH global step (12 points)
_UMH_HEX_LARGE = [
    (-4, 0), (-3, -2), (-2, -4), (0, -4), (2, -4), (3, -2),
    (4,  0), (3,  2),  (2,  4),  (0,  4), (-2, 4), (-3, 2),
]

# 4-point diamond (used in sub-pel refinement)
_DIAMOND_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# 8-point full diamond (used in half/qpel refinement)
_DIAMOND_8 = [
    (-1, -1), (0, -1), (1, -1),
    (-1,  0),          (1,  0),
    (-1,  1), (0,  1), (1,  1),
]

# Hadamard 4×4 kernel (for SATD)
_H4 = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1],
], dtype=np.int32)


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class MotionVector:
    """Best motion vector found for one block."""
    mvx:        int    # horizontal MV in quarter-pixel units (positive = right)
    mvy:        int    # vertical   MV in quarter-pixel units (positive = down)
    sad_cost:   int    # SAD of the best integer-pel candidate
    satd_cost:  int    # SATD after sub-pel refinement (0 if refinement skipped)
    ref_idx:    int    # reference frame index (always 0 for single-ref ME)
    algorithm:  str    # search algorithm used

    # ── Convenience ──────────────────────────────────────────────────────
    @property
    def mvx_int(self) -> int:
        """Integer-pixel horizontal displacement."""
        return self.mvx // 4

    @property
    def mvy_int(self) -> int:
        """Integer-pixel vertical displacement."""
        return self.mvy // 4

    @property
    def is_zero_mv(self) -> bool:
        return self.mvx == 0 and self.mvy == 0

    def __repr__(self) -> str:
        return (f"MotionVector(mv=({self.mvx/4:+.2f}, {self.mvy/4:+.2f}) px, "
                f"SAD={self.sad_cost}, SATD={self.satd_cost})")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_motion(
    block:          np.ndarray,
    ref_frame:      np.ndarray,
    origin:         tuple[int, int] = (0, 0),
    search_range:   int = DEFAULT_SEARCH_RANGE,
    algorithm:      Literal["full", "hex", "umh"] = ALGO_HEX,
    use_satd_refine: bool = True,
    ref_idx:        int = 0,
) -> MotionVector:
    """
    Find the best motion vector for a luma block in a reference frame.

    Three-stage pipeline:
        1. Integer-pel search  (SAD, chosen algorithm)
        2. Half-pel refinement (SATD, 8-point diamond)
        3. Quarter-pel refinement (SATD, 8-point diamond)

    Parameters
    ----------
    block : np.ndarray
        Current luma block to match, shape (N, N), dtype uint8.
        N must be in {4, 8, 16, 32, 64}.
    ref_frame : np.ndarray
        Full reference luma frame, shape (H, W), dtype uint8.
        Must be large enough to contain the search window.
    origin : (x, y)
        Top-left corner of `block` in the reference frame (integer pixels).
        The search is centred here. x = column, y = row.
    search_range : int
        Integer-pixel search radius (default 32). Total window = (2R+N)×(2R+N).
    algorithm : {"full", "hex", "umh"}
        Integer-pel search strategy.
        "hex"  — fast hexagonal pattern (recommended, ~x265 default)
        "umh"  — uneven multi-hexagon (better quality, slower)
        "full" — exhaustive (for testing/reference only)
    use_satd_refine : bool
        Run half-pel and quarter-pel refinement steps (default True).
        Disable for speed benchmarking of integer-pel only.
    ref_idx : int
        Reference frame index stored in the returned MV (informational).

    Returns
    -------
    MotionVector
        Best MV in quarter-pixel units, with cost metrics.

    Raises
    ------
    ValueError
        If block size is unsupported, ref_frame is smaller than search window,
        or search_range is non-positive.
    """
    n = _validate_inputs(block, ref_frame, origin, search_range)
    ox, oy = origin

    # ── Stage 1: Integer-pel search ───────────────────────────────────────
    if algorithm == ALGO_FULL:
        best_ix, best_iy, best_sad = _search_full(block, ref_frame, ox, oy, search_range, n)
    elif algorithm == ALGO_HEX:
        best_ix, best_iy, best_sad = _search_hex(block, ref_frame, ox, oy, search_range, n)
    elif algorithm == ALGO_UMH:
        best_ix, best_iy, best_sad = _search_umh(block, ref_frame, ox, oy, search_range, n)
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: full, hex, umh.")

    if not use_satd_refine:
        return MotionVector(
            mvx=best_ix * 4, mvy=best_iy * 4,
            sad_cost=best_sad, satd_cost=0,
            ref_idx=ref_idx, algorithm=algorithm,
        )

    # ── Stage 2: Half-pel refinement ─────────────────────────────────────
    # Start at best integer position, search ±1 in 1/2-pel steps
    best_hx, best_hy, best_satd = _refine_subpel(
        block, ref_frame, ox, oy, best_ix, best_iy, n,
        step=2,           # 2 qpel units = 1 half-pel
        pattern=_DIAMOND_8,
    )

    # ── Stage 3: Quarter-pel refinement ──────────────────────────────────
    best_qx, best_qy, best_satd = _refine_subpel(
        block, ref_frame, ox, oy, best_hx, best_hy, n,
        step=1,           # 1 qpel unit = 1/4 pel
        pattern=_DIAMOND_8,
        start_in_qpel=True,
    )

    return MotionVector(
        mvx=best_qx, mvy=best_qy,
        sad_cost=best_sad, satd_cost=best_satd,
        ref_idx=ref_idx, algorithm=algorithm,
    )


def compute_sad(block: np.ndarray, ref_patch: np.ndarray) -> int:
    """
    Sum of Absolute Differences between a block and a reference patch.

    SAD = Σ |block[i,j] − ref[i,j]|

    Parameters
    ----------
    block     : (N, N) uint8 or int16 — current block
    ref_patch : (N, N) uint8 or int16 — reference candidate patch

    Returns
    -------
    int — SAD cost (non-negative)
    """
    return int(np.sum(np.abs(
        block.astype(np.int32) - ref_patch.astype(np.int32)
    )))


def compute_satd(block: np.ndarray, ref_patch: np.ndarray) -> int:
    """
    Sum of Absolute Hadamard-Transformed Differences.

    More accurate than SAD for sub-pel mode decision because it
    approximates transform-domain distortion. Applied in 4×4 sub-blocks.

    Parameters
    ----------
    block     : (N, N) uint8 or int16
    ref_patch : (N, N) uint8 or int16, same shape

    Returns
    -------
    int — SATD cost (non-negative)
    """
    residual = block.astype(np.int32) - ref_patch.astype(np.int32)
    n = residual.shape[0]
    total = 0
    for r in range(0, n, 4):
        for c in range(0, n, 4):
            sub = residual[r:r+4, c:c+4]
            h = _H4 @ sub @ _H4.T
            total += int(np.sum(np.abs(h)))
    return total >> 2


def interpolate_half_pel(
    ref_frame: np.ndarray,
    x: int,
    y: int,
    n: int,
) -> np.ndarray:
    """
    Extract a half-pixel interpolated N×N patch from the reference frame.

    Uses HEVC 6-tap luma filter (ISO/IEC 23008-2 §8.5.3.2.2).
    Coordinates (x, y) are in HALF-PIXEL units (2 = 1 integer pixel).

    Parameters
    ----------
    ref_frame : (H, W) uint8 — full reference frame
    x, y      : int          — top-left corner in half-pixel units
    n         : int          — block size

    Returns
    -------
    patch : (N, N) uint8 — interpolated patch, clipped to [0, 255]
    """
    # Convert half-pel coords to (integer pixel, fraction) pairs
    # Fraction: 0=integer, 1=half-pel offset
    ix, fx = x // 2, x % 2
    iy, fy = y // 2, y % 2

    return _interp_patch(ref_frame, ix, iy, fx * 4, fy * 4, n)


def interpolate_qpel(
    ref_frame: np.ndarray,
    qx: int,
    qy: int,
    n: int,
) -> np.ndarray:
    """
    Extract a quarter-pixel interpolated N×N patch from the reference frame.

    Coordinates (qx, qy) are in QUARTER-PIXEL units (4 = 1 integer pixel).

    Parameters
    ----------
    ref_frame : (H, W) uint8 — full reference frame
    qx, qy   : int           — top-left corner in quarter-pixel units
    n         : int           — block size

    Returns
    -------
    patch : (N, N) uint8 — interpolated patch
    """
    ix, fx = qx // 4, qx % 4   # integer part + fractional part (0-3)
    iy, fy = qy // 4, qy % 4
    return _interp_patch(ref_frame, ix, iy, fx * 2, fy * 2, n)


# ---------------------------------------------------------------------------
# Integer-pel search algorithms
# ---------------------------------------------------------------------------

def _search_full(
    block: np.ndarray,
    ref: np.ndarray,
    ox: int, oy: int,
    sr: int,
    n: int,
) -> tuple[int, int, int]:
    """Exhaustive full search over all integer candidates in ±sr range."""
    H, W = ref.shape
    best_sad = int(2**63 - 1)
    best_x, best_y = 0, 0

    for dy in range(-sr, sr + 1):
        for dx in range(-sr, sr + 1):
            cx, cy = ox + dx, oy + dy
            if cx < 0 or cy < 0 or cx + n > W or cy + n > H:
                continue
            sad = compute_sad(block, ref[cy:cy+n, cx:cx+n])
            if sad < best_sad:
                best_sad = sad
                best_x, best_y = dx, dy

    return best_x, best_y, best_sad


def _search_hex(
    block: np.ndarray,
    ref: np.ndarray,
    ox: int, oy: int,
    sr: int,
    n: int,
) -> tuple[int, int, int]:
    """
    Hexagonal search — x265-style fast integer ME.

    Two phases:
        Phase 1: Large hexagon steps until no improvement.
        Phase 2: 4-point small diamond refinement at the best position.

    Reduces candidates from O(sr²) to O(log sr) in typical video.
    """
    H, W = ref.shape

    def sad_at(dx: int, dy: int) -> int:
        cx, cy = ox + dx, oy + dy
        if cx < 0 or cy < 0 or cx + n > W or cy + n > H:
            return int(2**31)
        return compute_sad(block, ref[cy:cy+n, cx:cx+n])

    # Start at zero MV (temporal prediction)
    best_x, best_y = 0, 0
    best_sad = sad_at(0, 0)

    # Phase 1: large hexagon steps
    step = max(1, sr // 4)
    while step >= 1:
        moved = True
        while moved:
            moved = False
            for pdx, pdy in _HEX_PATTERN:
                dx = best_x + pdx * step
                dy = best_y + pdy * step
                if abs(dx) > sr or abs(dy) > sr:
                    continue
                s = sad_at(dx, dy)
                if s < best_sad:
                    best_sad, best_x, best_y = s, dx, dy
                    moved = True
        step //= 2

    # Phase 2: 4-point diamond refinement
    improved = True
    while improved:
        improved = False
        for pdx, pdy in _DIAMOND_4:
            dx, dy = best_x + pdx, best_y + pdy
            if abs(dx) > sr or abs(dy) > sr:
                continue
            s = sad_at(dx, dy)
            if s < best_sad:
                best_sad, best_x, best_y = s, dx, dy
                improved = True

    return best_x, best_y, best_sad


def _search_umh(
    block: np.ndarray,
    ref: np.ndarray,
    ox: int, oy: int,
    sr: int,
    n: int,
) -> tuple[int, int, int]:
    """
    Uneven Multi-Hexagon search — higher quality than hex, slower.

    Three phases:
        Phase 1: Coarse global scan — test a sparse grid at large steps.
        Phase 2: Large hexagon around best coarse candidate until no gain.
        Phase 3: 4-point diamond fine refinement.
    """
    H, W = ref.shape

    def sad_at(dx: int, dy: int) -> int:
        cx, cy = ox + dx, oy + dy
        if cx < 0 or cy < 0 or cx + n > W or cy + n > H:
            return int(2**31)
        return compute_sad(block, ref[cy:cy+n, cx:cx+n])

    best_x, best_y = 0, 0
    best_sad = sad_at(0, 0)

    # Phase 1: sparse coarse grid (step = sr//4, covers global range)
    step = max(4, sr // 4)
    for dy in range(-sr, sr + 1, step):
        for dx in range(-sr, sr + 1, step):
            s = sad_at(dx, dy)
            if s < best_sad:
                best_sad, best_x, best_y = s, dx, dy

    # Phase 2: large hex around best coarse position
    for scale in [4, 2, 1]:
        moved = True
        while moved:
            moved = False
            for pdx, pdy in _UMH_HEX_LARGE:
                dx = best_x + pdx * scale
                dy = best_y + pdy * scale
                if abs(dx) > sr or abs(dy) > sr:
                    continue
                s = sad_at(dx, dy)
                if s < best_sad:
                    best_sad, best_x, best_y = s, dx, dy
                    moved = True

    # Phase 3: fine diamond
    improved = True
    while improved:
        improved = False
        for pdx, pdy in _DIAMOND_4:
            dx, dy = best_x + pdx, best_y + pdy
            if abs(dx) > sr or abs(dy) > sr:
                continue
            s = sad_at(dx, dy)
            if s < best_sad:
                best_sad, best_x, best_y = s, dx, dy
                improved = True

    return best_x, best_y, best_sad


# ---------------------------------------------------------------------------
# Sub-pixel refinement
# ---------------------------------------------------------------------------

def _refine_subpel(
    block:          np.ndarray,
    ref:            np.ndarray,
    ox:             int,          # block origin x in the reference frame (integer pixels)
    oy:             int,          # block origin y in the reference frame (integer pixels)
    start_x:        int,
    start_y:        int,
    n:              int,
    step:           int,
    pattern:        list[tuple[int, int]],
    start_in_qpel:  bool = False,
) -> tuple[int, int, int]:
    """
    Sub-pixel refinement around a starting MV position.

    start_x/y are MV displacements relative to origin (ox, oy).
    Internally we maintain MVs in qpel units and convert to absolute
    frame coordinates when calling _get_qpel_patch.

    step=2 → half-pel refinement (±1 half-pel step)
    step=1 → quarter-pel refinement (±1 qpel step)
    """
    # Convert MV displacement to qpel units
    if start_in_qpel:
        qx, qy = start_x, start_y
    else:
        qx, qy = start_x * 4, start_y * 4

    def satd_at(mv_qpx: int, mv_qpy: int) -> int:
        # Absolute qpel position = origin (in qpel) + MV (in qpel)
        abs_qx = ox * 4 + mv_qpx
        abs_qy = oy * 4 + mv_qpy
        patch = _get_qpel_patch(ref, abs_qx, abs_qy, n)
        if patch is None:
            return int(2**31)
        return compute_satd(block, patch)

    best_qx, best_qy = qx, qy
    best_satd = satd_at(qx, qy)

    improved = True
    while improved:
        improved = False
        for pdx, pdy in pattern:
            cqx = best_qx + pdx * step
            cqy = best_qy + pdy * step
            s = satd_at(cqx, cqy)
            if s < best_satd:
                best_satd, best_qx, best_qy = s, cqx, cqy
                improved = True

    return best_qx, best_qy, best_satd


def _get_qpel_patch(
    ref: np.ndarray,
    qx: int,
    qy: int,
    n: int,
) -> np.ndarray | None:
    """
    Extract an interpolated N×N patch at quarter-pixel position (qx, qy).

    qx, qy are absolute quarter-pixel coordinates in the reference frame
    (not MV offsets). Returns None if out of bounds.
    """
    ix, fx = qx // 4, qx % 4
    iy, fy = qy // 4, qy % 4

    H, W = ref.shape
    # 8-tap filter starts reading 3 pixels before ix/iy (FILTER_OFFSET=3)
    # and reads up to n+7 beyond that start. Guard the full range.
    if ix - 3 < 0 or iy - 3 < 0 or ix + n + 4 > W or iy + n + 4 > H:
        return None

    patch = _interp_patch(ref, ix, iy, fx * 2, fy * 2, n)
    return patch


def _interp_patch(
    ref: np.ndarray,
    ix: int,
    iy: int,
    frac_x: int,
    frac_y: int,
    n: int,
) -> np.ndarray:
    """
    2-D separable interpolation using HEVC 6-tap luma filter.

    frac_x, frac_y are fractional offsets in EIGHTH-pixel units (0..7).
        frac=0 → integer pel (copy)
        frac=4 → half-pel
        frac=2 → quarter-pel
        frac=6 → three-quarter-pel

    Two-pass: horizontal first, then vertical.
    """
    H, W = ref.shape

    # The HEVC 8-tap luma filter has its identity tap at index 3:
    #   [0, 0, 0, 64, 0, 0, 0, 0]  ← frac=0, dominant at position 3
    # Row sampling must start at ix-3 so that the sum over taps 0..7
    # centred at tap 3 maps exactly to pixel ix+col for integer-pel.
    FILTER_OFFSET = 3   # index of the identity/dominant tap
    extra = 7           # tap-1, samples needed beyond block for 8-tap filter

    h_coeffs = _get_filter_coeffs(frac_x)
    v_coeffs = _get_filter_coeffs(frac_y)
    tap = len(h_coeffs)
    rows_needed = n + extra

    def get_row(y_idx: int, x_start: int) -> np.ndarray:
        """Clamp-extended row fetch starting at x_start, length n+extra."""
        y_idx = max(0, min(y_idx, H - 1))
        xs = np.arange(x_start, x_start + n + extra)
        xs = np.clip(xs, 0, W - 1)
        return ref[y_idx, xs].astype(np.int32)

    # Horizontal pass: fetch rows starting at iy-FILTER_OFFSET, x at ix-FILTER_OFFSET
    h_rows = np.zeros((rows_needed, n), dtype=np.int32)
    for i in range(rows_needed):
        row = get_row(iy - FILTER_OFFSET + i, ix - FILTER_OFFSET)
        for col in range(n):
            acc = sum(h_coeffs[t] * int(row[col + t]) for t in range(tap))
            h_rows[i, col] = acc

    # Round horizontal result (6-bit shift per HEVC spec)
    h_rows = (h_rows + 32) >> 6

    # Vertical pass: filter each output column, starting rows at FILTER_OFFSET
    patch = np.zeros((n, n), dtype=np.int32)
    for col in range(n):
        for row in range(n):
            acc = sum(v_coeffs[t] * int(h_rows[row + t, col]) for t in range(tap))
            patch[row, col] = (acc + 32) >> 6

    return np.clip(patch, 0, 255).astype(np.uint8)


def _get_filter_coeffs(frac: int) -> list[int]:
    """
    Return 8-tap HEVC luma filter coefficients for a given fractional offset
    in eighth-pixel units (0..7).

    The HEVC luma filter is symmetric: the filter for fraction f and fraction
    (8-f) are reverses of each other.

        frac=0 → [0,0,0,64,0,0,0,0]      integer pel (identity)
        frac=2 → [-1,4,-11,40,40,-11,4,-1] 1/4-pel
        frac=4 → [0,-3,9,20,20,9,-3,0]    1/2-pel (symmetric)
        frac=6 → [-1,4,-11,40,40,-11,4,-1][::-1]  3/4-pel = reverse of 1/4
    """
    frac = frac % 8   # safety clamp to [0, 7]
    if frac <= 4:
        return _HEVC_LUMA_FILTER[frac]
    else:
        # frac=5 → reverse of frac=3, frac=6 → reverse of frac=2, etc.
        return _HEVC_LUMA_FILTER[8 - frac][::-1]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(
    block:       np.ndarray,
    ref_frame:   np.ndarray,
    origin:      tuple[int, int],
    search_range: int,
) -> int:
    """Validate all inputs. Returns block size N."""
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError(
            f"Block must be a square 2-D array, got shape {block.shape}."
        )
    n = block.shape[0]
    if n not in VALID_BLOCK_SIZES:
        raise ValueError(
            f"Unsupported block size {n}. Supported: {VALID_BLOCK_SIZES}."
        )
    if ref_frame.ndim != 2:
        raise ValueError(
            f"ref_frame must be a 2-D array (luma plane), got {ref_frame.ndim}-D."
        )
    if search_range <= 0:
        raise ValueError(f"search_range must be positive, got {search_range}.")

    H, W = ref_frame.shape
    ox, oy = origin
    # Check the reference frame is large enough for at least a zero-MV patch
    if ox < 0 or oy < 0 or ox + n > W or oy + n > H:
        raise ValueError(
            f"Origin ({ox}, {oy}) + block size {n} is outside "
            f"reference frame {W}×{H}."
        )
    return n