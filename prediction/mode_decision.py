"""
mode_decision.py — HEVC Rate-Distortion Mode Decision
The RDO arbiter that decides between Intra and Inter prediction per CU.

Pipeline position
-----------------
    intra_estimation.py  ─┐
    intra_prediction.py   ├──▶  [mode_decision.py]  ──▶  cabac.py
    motion_estimation.py  │      (winner CU)               (entropy coding)
    motion_compensation.py┘

What this module does
----------------------
Every Coding Unit (CU) in HEVC must be encoded as one of:

    INTRA  — predict from reconstructed neighbours (I-slices and I-blocks)
    INTER  — predict from a reference frame using a motion vector (P/B-slices)
    SKIP   — zero residual + MV inherited from spatial/temporal neighbours

The decision is made using Lagrangian Rate-Distortion Optimisation (RDO):

    J(mode) = D(mode) + λ · R(mode)

    D — distortion (SSE between original and reconstructed block)
    R — rate       (bits needed to encode this mode)
    λ — Lagrange multiplier, derived from QP

The mode with minimum J wins.

Lagrange multiplier
--------------------
HEVC encoders use:  λ = 0.57 · 2^((QP − 12) / 3)

This is the x265 formula. For the golden model we use the same formula.
Rate is approximated by counting non-zero quantised coefficients (proxy
for entropy-coded bit cost) because we have no CABAC arithmetic yet.

Rate proxy formula
------------------
For mode decision we approximate rate as:

    R ≈ nnz(quant_levels)       non-zero coefficient count
      + mode_bits               mode signalling overhead

Mode bits (rough approximation):
    Intra:  3 bits  (mode index is CABAC-coded, ~3 bits average)
    Inter:  5 bits  (MV delta + reference index)
    Skip:   1 bit   (single skip flag)

Public API
----------
    decide_mode(block, ref_above, ref_left, ref_frame, origin,
                qp, slice_type, search_range)  -> ModeDecision
    compute_lambda(qp)                          -> float
    rd_cost(distortion, rate, lambda_)          -> float
    compute_sse(original, pred)                 -> int
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from prediction.intra_estimation import estimate_intra_mode, VALID_BLOCK_SIZES
from prediction.intra_prediction import full_intra_pipeline, IntraResult
from prediction.motion_estimation import estimate_motion, MotionVector, ALGO_HEX
from prediction.motion_compensation import full_inter_pipeline, InterResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLICE_I = "I"   # intra-only — only intra modes considered
SLICE_P = "P"   # uni-directional inter — intra and inter both considered
SLICE_B = "B"   # bi-directional inter  — not fully implemented in golden model

# Mode identifiers in ModeDecision
MODE_INTRA = "intra"
MODE_INTER = "inter"
MODE_SKIP  = "skip"

# Approximate mode signalling cost in bits (used in rate proxy)
_MODE_BITS: dict[str, int] = {
    MODE_INTRA: 3,
    MODE_INTER: 5,
    MODE_SKIP:  1,
}

# Skip detection threshold: if inter SAD < this, treat as skip candidate
_SKIP_SAD_THRESHOLD: int = 4


# ---------------------------------------------------------------------------
# Public return type
# ---------------------------------------------------------------------------

@dataclass
class ModeDecision:
    """Complete RDO result for one Coding Unit."""
    mode:         str             # "intra", "inter", or "skip"
    rd_cost:      float           # winning J = D + λ·R
    distortion:   int             # SSE between original and prediction
    rate_proxy:   int             # non-zero coeffs + mode bits
    qp:           int             # QP used for this decision
    lambda_:      float           # Lagrange multiplier

    # Intra fields (populated when mode == "intra")
    intra_mode:   int | None            = None
    intra_result: IntraResult | None    = None

    # Inter fields (populated when mode == "inter" or "skip")
    mv:           MotionVector | None   = None
    inter_result: InterResult | None    = None

    @property
    def is_intra(self) -> bool:  return self.mode == MODE_INTRA
    @property
    def is_inter(self) -> bool:  return self.mode == MODE_INTER
    @property
    def is_skip(self)  -> bool:  return self.mode == MODE_SKIP

    @property
    def pred(self) -> np.ndarray | None:
        """Convenience: return the winning prediction block."""
        if self.intra_result is not None:
            return self.intra_result.pred
        if self.inter_result is not None:
            return self.inter_result.pred
        return None

    @property
    def residual(self) -> np.ndarray | None:
        if self.intra_result is not None:
            return self.intra_result.residual
        if self.inter_result is not None:
            return self.inter_result.residual
        return None

    @property
    def quant_levels(self) -> np.ndarray | None:
        if self.intra_result is not None:
            return self.intra_result.quant_levels
        if self.inter_result is not None:
            return self.inter_result.quant_levels
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decide_mode(
    block:        np.ndarray,
    ref_above:    np.ndarray,
    ref_left:     np.ndarray,
    ref_frame:    np.ndarray | None,
    origin:       tuple[int, int],
    qp:           int = 28,
    slice_type:   Literal["I", "P", "B"] = SLICE_P,
    search_range: int = 32,
    rmd_candidates: int = 8,
) -> ModeDecision:
    """
    RDO mode decision for one Coding Unit.

    Evaluates intra (always) and inter (for P/B slices) candidates,
    returning the mode with minimum Lagrangian cost J = D + λ·R.

    Parameters
    ----------
    block : np.ndarray
        Original luma pixel block, shape (N, N), dtype uint8.
    ref_above : np.ndarray
        Intra reference samples above the block, shape (2N+1,), int16.
    ref_left : np.ndarray
        Intra reference samples left of the block, shape (2N+1,), int16.
    ref_frame : np.ndarray or None
        Full luma reference frame for inter prediction, shape (H, W), uint8.
        None forces intra-only mode (same as slice_type="I").
    origin : (ox, oy)
        Top-left corner of the block in the reference frame (integer pixels).
    qp : int
        Quantisation parameter [0, 51].
    slice_type : {"I", "P", "B"}
        "I" → intra-only. "P" → intra + inter. "B" → same as P (golden model).
    search_range : int
        ME integer-pel search radius (default 32).
    rmd_candidates : int
        Number of RMD shortlist candidates for intra estimation (default 8).

    Returns
    -------
    ModeDecision
        Contains the winning mode, RD cost, prediction, residual, and
        quantised levels ready for entropy coding.

    Raises
    ------
    ValueError
        If block size is unsupported or QP is out of range.
    """
    n = _validate_block(block, qp)
    lambda_ = compute_lambda(qp)

    # ── Intra candidate ───────────────────────────────────────────────────
    intra_candidate = _evaluate_intra(
        block, ref_above, ref_left, qp, lambda_, rmd_candidates
    )

    # ── Inter candidate (P/B slices with ref frame available) ─────────────
    inter_candidate = None
    if slice_type in (SLICE_P, SLICE_B) and ref_frame is not None:
        inter_candidate = _evaluate_inter(
            block, ref_frame, origin, qp, lambda_, search_range
        )

    # ── Skip candidate (zero residual, inherit MV) ────────────────────────
    skip_candidate = None
    if inter_candidate is not None:
        skip_candidate = _evaluate_skip(
            block, ref_frame, origin, inter_candidate, qp, lambda_
        )

    # ── Pick winner by RD cost ────────────────────────────────────────────
    candidates = [c for c in [intra_candidate, inter_candidate, skip_candidate]
                  if c is not None]
    winner = min(candidates, key=lambda c: c.rd_cost)
    return winner


def compute_lambda(qp: int) -> float:
    """
    Lagrange multiplier for RDO at a given QP.

    Formula: λ = 0.57 · 2^((QP − 12) / 3)

    This is the standard x265 / HM formula. λ scales with quantisation
    step size: higher QP → coarser quantisation → larger λ → rate is
    worth less distortion → encoder accepts more distortion to save bits.

    Parameters
    ----------
    qp : int — quantisation parameter [0, 51]

    Returns
    -------
    float — Lagrange multiplier λ > 0
    """
    return 0.57 * (2.0 ** ((qp - 12) / 3.0))


def rd_cost(distortion: int | float, rate: int | float, lambda_: float) -> float:
    """
    Lagrangian RD cost: J = D + λ · R.

    Parameters
    ----------
    distortion : int or float — SSE between original and reconstruction
    rate       : int or float — bit-count proxy (non-zero coefficients + mode bits)
    lambda_    : float        — Lagrange multiplier from compute_lambda()

    Returns
    -------
    float — J value (lower is better)
    """
    return float(distortion) + lambda_ * float(rate)


def compute_sse(original: np.ndarray, pred: np.ndarray) -> int:
    """
    Sum of Squared Errors between original and prediction.

    SSE = Σ (original[i,j] − pred[i,j])²

    Used as the distortion metric D in J = D + λ·R.

    Parameters
    ----------
    original : np.ndarray — original pixel block (uint8 or int16)
    pred     : np.ndarray — prediction block, same shape (uint8)

    Returns
    -------
    int — SSE (non-negative)
    """
    diff = original.astype(np.int32) - pred.astype(np.int32)
    return int(np.sum(diff ** 2))


# ---------------------------------------------------------------------------
# Internal candidate evaluators
# ---------------------------------------------------------------------------

def _evaluate_intra(
    block:      np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    qp:         int,
    lambda_:    float,
    rmd_candidates: int,
) -> ModeDecision:
    """Run RMD + RDO intra search and return a ModeDecision candidate."""
    # Stage 1: find best intra mode via SAD/SATD cost
    best = estimate_intra_mode(
        block, ref_above, ref_left,
        rmd_candidates=rmd_candidates,
    )

    # Stage 2: run the full pipeline with the winning mode
    result = full_intra_pipeline(
        block, best.mode, ref_above, ref_left,
        qp=qp, is_intra=True,
    )

    sse    = compute_sse(block, result.pred)
    nnz    = int(np.count_nonzero(result.quant_levels))
    rate   = nnz + _MODE_BITS[MODE_INTRA]
    j_cost = rd_cost(sse, rate, lambda_)

    return ModeDecision(
        mode=MODE_INTRA,
        rd_cost=j_cost,
        distortion=sse,
        rate_proxy=rate,
        qp=qp,
        lambda_=lambda_,
        intra_mode=best.mode,
        intra_result=result,
    )


def _evaluate_inter(
    block:        np.ndarray,
    ref_frame:    np.ndarray,
    origin:       tuple[int, int],
    qp:           int,
    lambda_:      float,
    search_range: int,
) -> ModeDecision | None:
    """Run ME + MC and return an inter ModeDecision candidate.

    Returns None if the winning MV places the compensation patch outside
    the reference frame bounds (can happen when the frame is small relative
    to the search range or origin is near the frame edge).
    """
    mv = estimate_motion(
        block, ref_frame, origin,
        search_range=search_range,
        algorithm=ALGO_HEX,
        use_satd_refine=True,
    )

    try:
        result = full_inter_pipeline(
            block, ref_frame, origin, mv, qp=qp
        )
    except ValueError:
        # MV is out of bounds for this frame — inter not available
        return None

    sse    = compute_sse(block, result.pred)
    nnz    = int(np.count_nonzero(result.quant_levels))
    rate   = nnz + _MODE_BITS[MODE_INTER]
    j_cost = rd_cost(sse, rate, lambda_)

    return ModeDecision(
        mode=MODE_INTER,
        rd_cost=j_cost,
        distortion=sse,
        rate_proxy=rate,
        qp=qp,
        lambda_=lambda_,
        mv=mv,
        inter_result=result,
    )


def _evaluate_skip(
    block:          np.ndarray,
    ref_frame:      np.ndarray,
    origin:         tuple[int, int],
    inter_decision: ModeDecision,
    qp:             int,
    lambda_:        float,
) -> ModeDecision | None:
    """
    Evaluate SKIP mode — zero MV, zero residual.

    In HEVC, SKIP means the MV is *inherited* from a merge candidate
    (typically (0,0) in the golden model since we have no merge list).
    No MV is explicitly signalled and no residual is coded.
    Rate = 1 bit (skip flag only).

    SKIP is evaluated using MV=(0,0) exclusively. If the inter candidate
    found a non-zero MV, that is INTER-with-zero-residual, not SKIP.
    Only evaluate SKIP when the zero-MV prediction is itself a good match.
    """
    if inter_decision.inter_result is None:
        return None

    # SKIP prediction uses MV=(0,0): copy directly from reference at origin
    ox, oy = origin
    n = block.shape[0]
    H, W = ref_frame.shape
    if oy + n > H or ox + n > W:
        return None

    skip_pred = ref_frame[oy:oy+n, ox:ox+n].copy()   # zero-MV patch
    sse    = compute_sse(block, skip_pred)
    rate   = _MODE_BITS[MODE_SKIP]                    # 1 bit — skip flag only
    j_cost = rd_cost(sse, rate, lambda_)

    from prediction.motion_compensation import InterResult  # noqa: F401 — already at top level
    return ModeDecision(
        mode=MODE_SKIP,
        rd_cost=j_cost,
        distortion=sse,
        rate_proxy=rate,
        qp=qp,
        lambda_=lambda_,
        mv=MotionVector(0, 0, 0, 0, inter_decision.mv.ref_idx, "skip"),
        inter_result=InterResult(
            mv=MotionVector(0, 0, 0, 0, inter_decision.mv.ref_idx, "skip"),
            pred=skip_pred,
            residual=np.zeros((n, n), dtype=np.int16),
            dct_coeffs=None,
            quant_levels=np.zeros((n, n), dtype=np.int32),
        ),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_block(block: np.ndarray, qp: int) -> int:
    if block.ndim != 2 or block.shape[0] != block.shape[1]:
        raise ValueError(
            f"Block must be a square 2-D array, got shape {block.shape}."
        )
    n = block.shape[0]
    if n not in VALID_BLOCK_SIZES:
        raise ValueError(
            f"Unsupported block size {n}. Supported: {VALID_BLOCK_SIZES}."
        )
    if not (0 <= qp <= 51):
        raise ValueError(f"QP must be in [0, 51], got {qp}.")
    return n