"""
tu_split.py — HEVC Transform Unit Recursive Split
Decides how to split the residual of a leaf CU into Transform Units (TUs).

Pipeline position
-----------------
    cu_split.py  →  mode_decision.py  →  [tu_split.py]  →  cabac.py
    (CU leaf          (pred pixels,         (TU tree,          (entropy
     decided)          residual)             coefficients)       coding)

Relationship to cu_split.py
-----------------------------
    cu_split.py  — decides the CU quad-tree (how to partition 64×64 into CUs)
    tu_split.py  — decides the TU quad-tree (how to partition each CU's residual)

The TU tree is INDEPENDENT from the CU tree:
    - The CU tree controls which prediction mode to use
    - The TU tree controls how to transform the residual from that prediction
    - A CU of size N×N can have TUs of size N, N/2, N/4, or N/8

Why split the TU separately from the CU?
------------------------------------------
The residual of a CU may not be uniform. Consider a 32×32 CU predicted by
a DC mode: the prediction is perfect in the centre but has errors near the
block boundary. Splitting the TU lets the encoder concentrate bits on the
error-prone regions while using a coarser (cheaper) transform elsewhere.

HEVC TU split rules (ISO/IEC 23008-2 §7.3.8.8)
-------------------------------------------------
    - Max TU size: min(CU_size, 32×32) — TUs cannot exceed 32×32
    - Min TU size: 4×4
    - TU depth relative to CU: 0 (no split) to max_tu_depth (default 3)
    - Intra: TU must have same size as PU for 4×4 and 8×8 CUs (implicit split disabled)
    - Inter: TU split is independent of PU (residual split is flexible)
    - The 4×4 DST-7 applies only to 4×4 intra luma TUs

Cost model
-----------
The TU split decision uses Lagrangian RDO, same formula as CU split:

    J = D + λ · R

    D = SSE(residual, reconstructed_residual)  — transform-domain distortion
    R = nnz(quantised_coefficients)            — rate proxy

Leaf TU cost:
    Forward DCT → quantise → dequantise → IDCT
    D = SSE(orig_residual, recon_residual)
    R = count_nonzero(levels)

Split TU cost:
    Σ J(child_TU) for 4 quadrant sub-residuals

Data structures
---------------
TU nodes reuse QuadNode from quad_tree.py. The TU tree root is a new
QuadNode whose x, y, size match the CU, but whose pixel data fields
(residual, coeffs, levels, recon) are populated at the TU level.

Public API
----------
    split_tu(residual, cu_x, cu_y, cu_size, qp, pred_mode,
             max_tu_depth, min_tu_size)   -> TUResult
    encode_tu_leaf(residual, qp, pred_mode, use_dst)
                                          -> TULeafResult
    compute_tu_rd_cost(residual, qp, lambda_, pred_mode, use_dst)
                                          -> (cost, leaf_result)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Literal

from quad_tree import QuadNode, NodeState, PredMode, MIN_CU_SIZE
from transform.dct import forward_dct, forward_dst
from transform.quantizer import quantize, dequantize
from transform.idct import inverse_dct, inverse_dst
from prediction.mode_decision import compute_lambda

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TU_SIZE: int = 32   # HEVC luma max TU size
MIN_TU_SIZE: int = 4    # HEVC luma min TU size
MAX_TU_DEPTH: int = 3   # max TU splits relative to CU (default in HEVC Main)

VALID_TU_SIZES = (32, 16, 8, 4)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class TULeafResult:
    """Result of encoding one leaf TU (no further split)."""
    x:         int             # top-left column in frame pixels
    y:         int             # top-left row in frame pixels
    size:      int             # TU width = height
    residual:  np.ndarray      # original residual, (size, size), int16
    coeffs:    np.ndarray      # DCT/DST coefficients, (size, size), int32
    levels:    np.ndarray      # quantised levels, (size, size), int32
    recon_res: np.ndarray      # reconstructed residual after IQ+IDCT, (size, size), int16
    sse:       int             # SSE(residual, recon_res)
    nnz:       int             # non-zero coefficient count
    rd_cost:   float           # J = SSE + λ · nnz
    use_dst:   bool            # True if DST-7 was used (4×4 intra luma)


@dataclass
class TUResult:
    """Complete TU tree decision for one CU's residual."""
    root:      QuadNode        # TU quad-tree root node
    total_sse: int             # aggregate distortion across all TU leaves
    total_nnz: int             # aggregate non-zero coefficients
    rd_cost:   float           # total J = Σ J(leaf)
    leaves:    list[TULeafResult] = field(default_factory=list)

    @property
    def all_levels(self) -> list[np.ndarray]:
        """Ordered list of quantised level arrays (leaf raster order)."""
        return [lf.levels for lf in self.leaves]

    @property
    def all_recon(self) -> list[np.ndarray]:
        return [lf.recon_res for lf in self.leaves]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_tu(
    residual:    np.ndarray,
    cu_x:        int,
    cu_y:        int,
    cu_size:     int,
    qp:          int = 28,
    pred_mode:   str = "intra",
    max_tu_depth: int = MAX_TU_DEPTH,
    min_tu_size:  int = MIN_TU_SIZE,
) -> TUResult:
    """
    Recursively decide the TU split structure for one CU's residual.

    For each TU node, evaluates:
        J_leaf  = RD cost of encoding residual with one DCT of this size
        J_split = sum of J for four sub-residuals after splitting

    The winner is assigned to the TU quad-tree.

    Parameters
    ----------
    residual : np.ndarray
        Residual pixel block for this CU: original − prediction.
        Shape (cu_size, cu_size), dtype int16.
    cu_x, cu_y : int
        Top-left coordinate of the CU in the frame (for TU node geometry).
    cu_size : int
        CU size in pixels. Must be in {4, 8, 16, 32} for the transform pipeline.
    qp : int
        Quantisation parameter [0, 51].
    pred_mode : str
        "intra" or "inter" — affects DST-7 eligibility (4×4 intra luma uses DST-7).
    max_tu_depth : int
        Maximum number of recursive splits allowed relative to CU depth.
        Default 3 (HEVC Main profile).
    min_tu_size : int
        Minimum TU size in pixels (default 4).

    Returns
    -------
    TUResult — fully decided TU tree with all leaf results populated.

    Raises
    ------
    ValueError
        If residual shape doesn't match cu_size, or cu_size is unsupported.
    """
    _validate_residual(residual, cu_size)
    lambda_ = compute_lambda(qp)

    # Build TU quad-tree root node
    root = QuadNode(x=cu_x, y=cu_y, size=cu_size, depth=0)
    leaves: list[TULeafResult] = []

    def _recurse(
        node:    QuadNode,
        res:     np.ndarray,
        tu_depth: int,
    ) -> float:
        """
        Recursively decide TU split. Returns best J for this subtree.
        Populates `leaves` as a side effect.
        """
        n = node.size

        # ── HEVC §7.3.8.8: Intra 4×4 and 8×8 CUs — TU must equal PU size ──
        # For small intra CUs, the TU cannot be split below the CU size.
        # "Implicit split" is disabled for these sizes.
        if pred_mode == "intra" and cu_size <= 8:
            leaf = encode_tu_leaf(res, qp, pred_mode,
                                  use_dst=_use_dst(node.size, pred_mode))
            leaf.x = node.x; leaf.y = node.y; leaf.size = node.size
            _write_leaf_to_node(node, leaf)
            leaves.append(leaf)
            return leaf.rd_cost

        # ── Forced leaf: at minimum TU size or max depth ────────────────
        if n <= min_tu_size or tu_depth >= max_tu_depth or n > MAX_TU_SIZE:
            leaf = encode_tu_leaf(res, qp, pred_mode,
                                  use_dst=_use_dst(n, pred_mode))
            leaf.x = node.x; leaf.y = node.y; leaf.size = n
            _write_leaf_to_node(node, leaf)
            leaves.append(leaf)
            return leaf.rd_cost
        # ── Step 1: Evaluate as leaf (no split) ─────────────────────────
        j_leaf, leaf_result = compute_tu_rd_cost(
            res, qp, lambda_, pred_mode, use_dst=_use_dst(n, pred_mode)
        )
        # Set the geometry on the result returned by encode_tu_leaf (x/y/size default to 0)
        leaf_result.x    = node.x
        leaf_result.y    = node.y
        leaf_result.size = n
        leaf_snap = _tu_leaf_snapshot(leaf_result)

        # ── Step 2: Evaluate as split ────────────────────────────────────
        hn = n // 2
        if hn < min_tu_size or hn not in VALID_TU_SIZES:
            # Cannot split further — force leaf
            _write_leaf_to_node(node, leaf_result)
            leaves.append(leaf_result)
            return j_leaf

        node_snap = (node.state, list(node.children))
        node.state = NodeState.UNSPLIT
        node.children = []
        children = node.split()

        # Extract 4 quadrant sub-residuals
        sub_res = [
            res[0:hn,  0:hn ].copy(),   # NW
            res[0:hn,  hn:n ].copy(),   # NE
            res[hn:n,  0:hn ].copy(),   # SW
            res[hn:n,  hn:n ].copy(),   # SE
        ]

        # Set correct frame-absolute positions for children leaves
        child_offsets = [(0,0),(hn,0),(0,hn),(hn,hn)]   # (dx, dy) within CU
        split_leaves_start = len(leaves)
        j_split = 0.0
        for child, sub, (dx, dy) in zip(children, sub_res, child_offsets):
            child.x = node.x + dx
            child.y = node.y + dy
            j_split += _recurse(child, sub, tu_depth + 1)

        # ── Step 3: Pick winner ──────────────────────────────────────────
        if j_split < j_leaf:
            node.rd_cost = j_split
        else:
            # Restore leaf — undo split, discard child leaves
            node.state, node.children = node_snap
            del leaves[split_leaves_start:]
            restored = _tu_leaf_restore(leaf_snap)
            _write_leaf_to_node(node, restored)
            leaves.append(restored)
            node.rd_cost = j_leaf
            j_split = j_leaf   # return value

        return node.rd_cost

    total_j = _recurse(root, residual, tu_depth=0)
    total_sse = sum(lf.sse for lf in leaves)
    total_nnz = sum(lf.nnz for lf in leaves)

    return TUResult(
        root=root,
        total_sse=total_sse,
        total_nnz=total_nnz,
        rd_cost=total_j,
        leaves=leaves,
    )


def encode_tu_leaf(
    residual:  np.ndarray,
    qp:        int,
    pred_mode: str = "intra",
    use_dst:   bool = False,
) -> TULeafResult:
    """
    Encode one leaf TU: Forward DCT/DST → Quantise → Dequantise → IDCT/IDST.

    This is the atomic unit of the transform pipeline. The result feeds
    into CABAC for entropy coding.

    Parameters
    ----------
    residual  : (N, N) int16 — residual block (original − prediction)
    qp        : int          — quantisation parameter [0, 51]
    pred_mode : str          — "intra" or "inter" (affects dead-zone width)
    use_dst   : bool         — use DST-7 instead of DCT-2 (4×4 intra luma)

    Returns
    -------
    TULeafResult (without x, y, size — caller fills those in)
    """
    n = residual.shape[0]
    is_intra = (pred_mode == "intra")
    lambda_  = compute_lambda(qp)

    # Forward transform
    if use_dst:
        coeffs = forward_dst(residual.astype(np.int32))
    else:
        coeffs = forward_dct(residual.astype(np.int32))

    # Quantisation
    levels = quantize(coeffs, qp=qp, is_intra=is_intra)

    # Inverse path (for distortion measurement)
    recoeffs = dequantize(levels, qp=qp)
    if use_dst:
        recon_res = inverse_dst(recoeffs)
    else:
        recon_res = inverse_dct(recoeffs)

    # Distortion and rate
    diff = residual.astype(np.int32) - recon_res.astype(np.int32)
    sse  = int(np.sum(diff ** 2))
    nnz  = int(np.count_nonzero(levels))
    cost = float(sse) + lambda_ * float(nnz)

    return TULeafResult(
        x=0, y=0, size=n,   # caller sets geometry
        residual=residual.astype(np.int16),
        coeffs=coeffs,
        levels=levels,
        recon_res=recon_res,
        sse=sse,
        nnz=nnz,
        rd_cost=cost,
        use_dst=use_dst,
    )


def compute_tu_rd_cost(
    residual:  np.ndarray,
    qp:        int,
    lambda_:   float,
    pred_mode: str = "intra",
    use_dst:   bool = False,
) -> tuple[float, TULeafResult]:
    """
    Compute the RD cost for encoding a residual block as a single TU.

    Convenience wrapper around encode_tu_leaf that returns (cost, result).

    Parameters
    ----------
    residual  : (N, N) int16
    qp        : int
    lambda_   : float — Lagrange multiplier (from compute_lambda)
    pred_mode : str
    use_dst   : bool

    Returns
    -------
    (rd_cost, TULeafResult)
    """
    result = encode_tu_leaf(residual, qp, pred_mode, use_dst)
    return result.rd_cost, result


def reconstruct_cu(
    pred:      np.ndarray,
    tu_result: TUResult,
    cu_x:      int,
    cu_y:      int,
    cu_size:   int,
) -> np.ndarray:
    """
    Reconstruct luma pixels for a CU from prediction + TU residuals.

    Assembles the reconstructed residual from all TU leaves back into
    a cu_size × cu_size block, then adds the prediction.

    Parameters
    ----------
    pred      : (cu_size, cu_size) uint8 — prediction block
    tu_result : TUResult — completed TU tree with all leaves
    cu_x, cu_y, cu_size : int — CU geometry

    Returns
    -------
    recon : (cu_size, cu_size) uint8 — reconstructed luma block, [0, 255]
    """
    recon_res = np.zeros((cu_size, cu_size), dtype=np.int16)

    for leaf in tu_result.leaves:
        lx = leaf.x - cu_x   # relative x within CU
        ly = leaf.y - cu_y   # relative y within CU
        n  = leaf.size
        recon_res[ly:ly+n, lx:lx+n] = leaf.recon_res

    recon = pred.astype(np.int16) + recon_res
    return np.clip(recon, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _use_dst(tu_size: int, pred_mode: str) -> bool:
    """True if DST-7 should be used: 4×4 intra luma only (HEVC §8.6.4)."""
    return tu_size == 4 and pred_mode == "intra"


def _validate_residual(residual: np.ndarray, cu_size: int) -> None:
    if residual.ndim != 2:
        raise ValueError(f"Residual must be 2-D, got {residual.ndim}-D.")
    if residual.shape != (cu_size, cu_size):
        raise ValueError(
            f"Residual shape {residual.shape} does not match cu_size {cu_size}."
        )
    if cu_size not in (4, 8, 16, 32):
        raise ValueError(
            f"Unsupported CU size {cu_size}. Supported: {{4, 8, 16, 32}}."
        )


def _write_leaf_to_node(node: QuadNode, leaf: TULeafResult) -> None:
    """Write TULeafResult fields into a QuadNode and mark it as LEAF."""
    node.residual = leaf.residual
    node.coeffs   = leaf.coeffs
    node.levels   = leaf.levels
    node.recon    = leaf.recon_res
    node.rd_cost  = leaf.rd_cost
    if node.state != NodeState.LEAF:
        node.state = NodeState.LEAF


def _tu_leaf_snapshot(leaf: TULeafResult) -> dict:
    """Snapshot all fields of a TULeafResult for O(1) restore."""
    return {k: v for k, v in vars(leaf).items()}


def _tu_leaf_restore(snap: dict) -> TULeafResult:
    """Restore a TULeafResult from a snapshot."""
    obj = TULeafResult.__new__(TULeafResult)
    for k, v in snap.items():
        setattr(obj, k, v)
    return obj