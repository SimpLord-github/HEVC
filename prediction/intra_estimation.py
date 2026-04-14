"""
intra_estimation.py — HEVC Intra-Picture Mode Estimation
Finds the best intra prediction mode for a given luma block.

Pipeline position
-----------------
    CTU splitter → [intra_estimation.py] → intra_prediction.py
                       (best mode index)     (generate pred pixels)

HEVC Intra Mode Set (ISO/IEC 23008-2 §8.4.4)
----------------------------------------------
HEVC defines 35 intra prediction modes for luma:

    Mode  0 — PLANAR       : weighted average of left column + top row + corner
    Mode  1 — DC            : mean of available reference samples
    Modes 2–34 — Angular    : 33 directional modes
                              Mode  2 = vertical-left  (135°)
                              Mode 18 = horizontal     (180°)
                              Mode 26 = vertical       (90°, straight up)
                              Mode 34 = vertical-right (45°)

Angular mode directions in degrees (angle from vertical):
    Modes 2–17 grow from -45° toward -1° (upper-left diagonal family)
    Mode 18   = pure horizontal (copy left column across)
    Modes 19–25 grow from +1° toward +25°
    Mode 26   = pure vertical   (copy top row down)
    Modes 27–34 grow from +28° toward +45°

Two-stage search strategy (mirrors hardware RDO pipeline)
----------------------------------------------------------
Stage 1 — Rough Mode Decision (RMD):
    Evaluate all 35 modes using SATD cost.
    Shortlist the top K candidates (default K=8).

Stage 2 — Rate-Distortion Optimisation (RDO):
    For each shortlisted candidate, compute full SAD cost.
    Return the mode with minimum cost.

Cost functions
--------------
SAD  (Sum of Absolute Differences):
    Pure distortion. Fast, used in final decision stage.

SATD (Sum of Absolute Hadamard-Transformed Differences):
    Transforms the residual via a 4×4 Hadamard before summing.
    Approximates the actual transform-domain cost without running a full
    DCT+Q pipeline. Used in the RMD shortlisting stage.

Public API
----------
    estimate_intra_mode(block, ref_above, ref_left, block_size) -> BestMode
    compute_sad(block, pred)                                     -> int
    compute_satd(block, pred)                                    -> int
    generate_all_costs(block, ref_above, ref_left, block_size)   -> dict[int, float]
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_INTRA_MODES: int = 35          # HEVC luma: modes 0–34
PLANAR_MODE:     int = 0
DC_MODE:         int = 1
ANGULAR_START:   int = 2
ANGULAR_END:     int = 34          # inclusive

# Supported block sizes for luma intra
VALID_BLOCK_SIZES = (4, 8, 16, 32)

# Default number of RMD shortlist candidates passed to the RDO stage
DEFAULT_RMD_CANDIDATES: int = 8

# Angular mode → intrinsic displacement (1/32 pixel units).
# Derived from HEVC spec Table 8-1. Positive = toward bottom-left,
# negative = toward upper-right.  Index = mode number (2..34).
# fmt: off
_ANGULAR_PARAMS: dict[int, int] = {
     2: -32,  3: -26,  4: -21,  5: -17,  6: -13,  7: -9,
     8:  -5,  9:  -2, 10:   0, 11:   2, 12:   5, 13:   9,
    14:  13, 15:  17, 16:  21, 17:  26, 18:  32,
    19:  26, 20:  21, 21:  17, 22:  13, 23:   9, 24:   5,
    25:   2, 26:   0, 27:  -2, 28:  -5, 29:  -9, 30: -13,
    31: -17, 32: -21, 33: -26, 34: -32,
}
# fmt: on

# Hadamard 4×4 kernel (unnormalised, integer)
_H4: np.ndarray = np.array([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1,  1, -1, -1],
    [1, -1, -1,  1],
], dtype=np.int32)


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class BestMode:
    """Result of intra mode estimation for one block."""
    mode:       int            # winning mode index [0, 34]
    sad_cost:   int            # SAD cost of the winning prediction
    satd_cost:  int            # SATD cost of the winning prediction
    candidates: list[int] = field(default_factory=list)  # RMD shortlist

    @property
    def is_angular(self) -> bool:
        return ANGULAR_START <= self.mode <= ANGULAR_END

    @property
    def is_dc(self) -> bool:
        return self.mode == DC_MODE

    @property
    def is_planar(self) -> bool:
        return self.mode == PLANAR_MODE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_intra_mode(
    block:      np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    block_size: Literal[4, 8, 16, 32] | None = None,
    rmd_candidates: int = DEFAULT_RMD_CANDIDATES,
) -> BestMode:
    """
    Find the best HEVC intra prediction mode for a luma block.

    Two-stage search:
        Stage 1 (RMD): SATD over all 35 modes → shortlist top K
        Stage 2 (RDO): SAD over shortlisted modes → return winner

    Parameters
    ----------
    block : np.ndarray
        Original luma block, shape (N, N), dtype uint8 or int16.
        Values represent pixel intensities before prediction.
    ref_above : np.ndarray
        Reference samples above the block. Shape (2*N+1,) in HEVC convention:
            [corner, top[0], top[1], ..., top[2N-1]]
        ref_above[0] is the top-left corner sample.
        ref_above[1:N+1] are the N samples directly above the block.
    ref_left : np.ndarray
        Reference samples to the left of the block. Shape (2*N+1,):
            [corner, left[0], left[1], ..., left[2N-1]]
        ref_left[0] is the same top-left corner as ref_above[0].
        ref_left[1:N+1] are the N samples directly to the left.
    block_size : {4, 8, 16, 32} or None
        If None, inferred from block.shape.
    rmd_candidates : int
        How many top-SATD modes to pass to the RDO stage (default 8).

    Returns
    -------
    BestMode
        Dataclass with mode index, SAD cost, SATD cost, and RMD shortlist.

    Raises
    ------
    ValueError
        If block is not square, size is unsupported, or reference arrays
        have incorrect length.

    Examples
    --------
    >>> import numpy as np
    >>> from prediction.intra_estimation import estimate_intra_mode
    >>> N = 8
    >>> block     = np.random.randint(0, 256, (N, N), dtype=np.uint8)
    >>> ref_above = np.full(2*N+1, 128, dtype=np.int16)
    >>> ref_left  = np.full(2*N+1, 128, dtype=np.int16)
    >>> result = estimate_intra_mode(block, ref_above, ref_left)
    >>> print(f"Best mode: {result.mode}  SAD: {result.sad_cost}")
    """
    n = _validate_inputs(block, ref_above, ref_left, block_size)
    block_i16 = block.astype(np.int16)

    # ── Stage 1: RMD — evaluate all 35 modes with SATD ────────────────────
    satd_costs: dict[int, int] = {}
    for mode in range(NUM_INTRA_MODES):
        pred = _predict(mode, ref_above, ref_left, n)
        satd_costs[mode] = compute_satd(block_i16, pred)

    # Shortlist: sort ascending by SATD, keep top K
    shortlist = sorted(satd_costs, key=satd_costs.__getitem__)[:rmd_candidates]

    # ── Stage 2: RDO — re-evaluate shortlist with SAD ─────────────────────
    best_mode, best_sad = -1, int(2**63 - 1)
    for mode in shortlist:
        pred = _predict(mode, ref_above, ref_left, n)
        sad  = compute_sad(block_i16, pred)
        if sad < best_sad:
            best_sad, best_mode = sad, mode

    best_pred  = _predict(best_mode, ref_above, ref_left, n)
    best_satd  = compute_satd(block_i16, best_pred)

    return BestMode(
        mode=best_mode,
        sad_cost=best_sad,
        satd_cost=best_satd,
        candidates=shortlist,
    )


def generate_all_costs(
    block:      np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    block_size: Literal[4, 8, 16, 32] | None = None,
) -> dict[int, dict[str, int]]:
    """
    Compute SAD and SATD costs for all 35 intra modes.

    Useful for visualising the mode cost landscape during debugging,
    or for feeding into an external RD optimiser in mode_decision.py.

    Returns
    -------
    dict mapping mode_index → {"sad": int, "satd": int}
    """
    n = _validate_inputs(block, ref_above, ref_left, block_size)
    block_i16 = block.astype(np.int16)

    costs: dict[int, dict[str, int]] = {}
    for mode in range(NUM_INTRA_MODES):
        pred = _predict(mode, ref_above, ref_left, n)
        costs[mode] = {
            "sad":  compute_sad(block_i16, pred),
            "satd": compute_satd(block_i16, pred),
        }
    return costs


def compute_sad(block: np.ndarray, pred: np.ndarray) -> int:
    """
    Sum of Absolute Differences between block and prediction.

    SAD = Σ |block[i,j] − pred[i,j]|

    Fast distortion metric. Used as the final RDO cost function in
    mode_decision.py (before full CABAC bit-counting is available).

    Parameters
    ----------
    block : np.ndarray — original pixel block (int16 or uint8)
    pred  : np.ndarray — prediction block, same shape

    Returns
    -------
    int — sum of absolute differences (non-negative)
    """
    return int(np.sum(np.abs(block.astype(np.int32) - pred.astype(np.int32))))


def compute_satd(block: np.ndarray, pred: np.ndarray) -> int:
    """
    Sum of Absolute Hadamard-Transformed Differences.

    SATD applies a 4×4 Hadamard transform to the residual before summing
    absolute values. This approximates the DCT-domain distortion without
    a full DCT+quantisation round-trip, making it ~3× more accurate than
    SAD for mode decisions at ~2× the compute cost.

    For blocks larger than 4×4, the Hadamard is applied to non-overlapping
    4×4 sub-blocks and the results are summed.

    Parameters
    ----------
    block : np.ndarray — original pixel block (int16 or uint8)
    pred  : np.ndarray — prediction block, same shape

    Returns
    -------
    int — SATD cost (non-negative integer)
    """
    residual = block.astype(np.int32) - pred.astype(np.int32)
    n = residual.shape[0]
    total = 0

    # Process in 4×4 sub-blocks
    for row in range(0, n, 4):
        for col in range(0, n, 4):
            sub = residual[row:row+4, col:col+4]
            # Two-pass Hadamard: H @ sub @ H^T
            h_rows = _H4 @ sub          # shape (4, 4)
            h_both = h_rows @ _H4.T     # shape (4, 4)
            total += int(np.sum(np.abs(h_both)))

    # Normalise: Hadamard of size 4 scales by 4 in each dimension → /4
    return total >> 2


# ---------------------------------------------------------------------------
# Intra prediction kernels (per HEVC spec §8.4.4)
# ---------------------------------------------------------------------------

def _predict(
    mode:      int,
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
    n:         int,
) -> np.ndarray:
    """Dispatch to the correct intra prediction kernel."""
    if mode == PLANAR_MODE:
        return _pred_planar(ref_above, ref_left, n)
    elif mode == DC_MODE:
        return _pred_dc(ref_above, ref_left, n)
    else:
        return _pred_angular(mode, ref_above, ref_left, n)


def _pred_planar(ref_above: np.ndarray, ref_left: np.ndarray, n: int) -> np.ndarray:
    """
    Planar prediction (mode 0) — ISO/IEC 23008-2 §8.4.4.2.4.

    Bilinear interpolation using the top row and left column.
    Each sample is a weighted blend of:
        - horizontal gradient from left reference
        - vertical gradient from top reference
        - bottom-right corner sample (ref_above[n] or ref_left[n])

    Formula per sample (x, y):
        pred[y, x] = ( (N-1-x)*ref_left[y+1] + (x+1)*ref_above[N] 
                     + (N-1-y)*ref_above[x+1] + (y+1)*ref_left[N] + N ) / (2N)
    """
    pred = np.zeros((n, n), dtype=np.int16)
    top  = ref_above[1:n+1].astype(np.int32)   # top[0..N-1]
    left = ref_left[1:n+1].astype(np.int32)    # left[0..N-1]
    top_right  = int(ref_above[n])              # top-right corner
    bot_left   = int(ref_left[n])              # bottom-left corner

    for y in range(n):
        for x in range(n):
            h = (n - 1 - x) * left[y] + (x + 1) * top_right
            v = (n - 1 - y) * top[x]  + (y + 1) * bot_left
            pred[y, x] = (h + v + n) >> (int(np.log2(n)) + 1)

    return pred


def _pred_dc(ref_above: np.ndarray, ref_left: np.ndarray, n: int) -> np.ndarray:
    """
    DC prediction (mode 1) — ISO/IEC 23008-2 §8.4.4.2.5.

    All samples set to the mean of available top and left reference samples.
    """
    top  = ref_above[1:n+1].astype(np.int32)
    left = ref_left[1:n+1].astype(np.int32)
    dc_val = int((np.sum(top) + np.sum(left) + n) >> (int(np.log2(n)) + 1))
    return np.full((n, n), dc_val, dtype=np.int16)


def _pred_angular(
    mode:      int,
    ref_above: np.ndarray,
    ref_left:  np.ndarray,
    n:         int,
) -> np.ndarray:
    """
    Angular prediction (modes 2–34) — ISO/IEC 23008-2 §8.4.4.2.6.

    Each mode projects reference samples at a fixed angle across the block.
    Modes 2–17 are derived from the left reference (horizontal family).
    Mode  18   is pure horizontal (copy left column rightward).
    Modes 19–25 blend both references.
    Mode  26   is pure vertical (copy top row downward).
    Modes 27–34 are derived from the top reference (vertical family).

    The intrinsic displacement d = _ANGULAR_PARAMS[mode] is in 1/32 pixel
    units. For each row/column, we compute the fractional offset and linearly
    interpolate between adjacent reference samples.
    """
    pred = np.zeros((n, n), dtype=np.int16)
    d    = _ANGULAR_PARAMS[mode]

    if mode > 18:
        # Vertical family (modes 19–34): project top reference downward
        ref = ref_above[1:2*n+1].astype(np.int32)
        for y in range(n):
            delta     = (y + 1) * d          # total displacement (1/32 px)
            base      = delta >> 5            # integer part
            frac      = delta & 31            # fractional part (0..31)
            for x in range(n):
                idx = x + base
                if frac == 0 or idx + 1 >= 2 * n:
                    s = _safe_ref(ref, idx)
                else:
                    s = (( (32 - frac) * _safe_ref(ref, idx)
                          + frac       * _safe_ref(ref, idx + 1) + 16) >> 5)
                pred[y, x] = np.clip(s, 0, 255)
    else:
        # Horizontal family: project left reference rightward
        ref = ref_left[1:2*n+1].astype(np.int32)
        for x in range(n):
            delta = (x + 1) * d
            base  = delta >> 5
            frac  = delta & 31
            for y in range(n):
                idx = y + base
                if frac == 0 or idx + 1 >= 2 * n:
                    s = _safe_ref(ref, idx)
                else:
                    s = (( (32 - frac) * _safe_ref(ref, idx)
                          + frac       * _safe_ref(ref, idx + 1) + 16) >> 5)
                pred[y, x] = np.clip(s, 0, 255)

    return pred


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_ref(ref: np.ndarray, idx: int) -> int:
    """Clamp-extend reference array access — mirrors hardware OOB handling."""
    idx = max(0, min(idx, len(ref) - 1))
    return int(ref[idx])


def _validate_inputs(
    block:      np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    block_size: int | None,
) -> int:
    """Validate all inputs and return block size N."""
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError(f"Block must be a square 2-D array, got shape {block.shape}.")
    n = block_size if block_size is not None else block.shape[0]
    if n not in VALID_BLOCK_SIZES:
        raise ValueError(f"Unsupported block size {n}. Supported: {VALID_BLOCK_SIZES}.")
    if block.shape[0] != n:
        raise ValueError(f"Block shape {block.shape} does not match block_size {n}.")

    expected_ref_len = 2 * n + 1
    if ref_above.ndim != 1 or len(ref_above) != expected_ref_len:
        raise ValueError(
            f"ref_above must be 1-D with length {expected_ref_len}, "
            f"got shape {ref_above.shape}."
        )
    if ref_left.ndim != 1 or len(ref_left) != expected_ref_len:
        raise ValueError(
            f"ref_left must be 1-D with length {expected_ref_len}, "
            f"got shape {ref_left.shape}."
        )
    return n