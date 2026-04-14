"""
cu_split.py — HEVC CU Recursive Split Decision
Decides how to recursively split a CTU into Coding Units using RDO.

Pipeline position
-----------------
    quad_tree.py  →  [cu_split.py]  →  mode_decision.py  →  tu_split.py
    (tree nodes)     (split/leaf      (pred mode per leaf)   (TU residual
                      decisions)                              split)

What this module does
----------------------
Given a QuadNode (initially the 64×64 CTU root), this module decides
whether to:

    A) Keep it as a LEAF CU and encode it with the best prediction mode, or
    B) SPLIT it into four children and recurse

The decision is made by comparing the RD cost of the two alternatives:

    J_leaf  = best_mode_cost(node)          — encode this node as one CU
    J_split = Σ J_best(child_i)  for i=0..3 — encode four sub-CUs

Whichever J is lower wins.

Split constraints (HEVC §7.3.8)
--------------------------------
    - Minimum CU size: 4×4 (set in quad_tree.MIN_CU_SIZE)
    - Maximum CU depth: 4 (log2(CTU_SIZE / MIN_CU_SIZE))
    - A node at MIN_CU_SIZE is always a leaf (forced)
    - A node at CTU_SIZE root is always evaluated for split

Partition modes (PU split within a leaf CU)
---------------------------------------------
Intra:  only PART_2Nx2N (the whole CU is one PU)
Inter:  PART_2Nx2N, PART_2NxN, PART_Nx2N, PART_NxN (8×8 only)

This file evaluates all eligible partition modes for each leaf candidate
and selects the best via RDO before comparing against split.

Context passed to mode_decision
---------------------------------
    ref_above, ref_left — intra reference samples for this CU position
    ref_frame           — full reference luma frame (for inter ME)
    origin              — (x, y) top-left of this CU in the frame

Public API
----------
    split_ctu(root, frame_luma, ref_above_fn, ref_left_fn,
              ref_frame, qp, slice_type, max_depth, min_size)
                                        -> QuadTree  (fully decided)
    evaluate_node(node, frame_luma, ref_above, ref_left,
                  ref_frame, qp, slice_type)
                                        -> float     (best J for this node)
    get_ref_samples(frame_luma, x, y, size) -> (ref_above, ref_left)
"""

from __future__ import annotations

import numpy as np
from typing import Literal

from partitioning.quad_tree import (
    QuadNode, QuadTree, NodeState, PredMode, PartMode,
    CTU_SIZE, MIN_CU_SIZE, MAX_CU_DEPTH, VALID_CU_SIZES,
)
from prediction.mode_decision import (
    decide_mode, compute_lambda, ModeDecision,
    MODE_INTRA, MODE_INTER, MODE_SKIP,
    SLICE_I, SLICE_P, SLICE_B,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Partition modes eligible per slice and block size
_INTRA_PART_MODES = [PartMode.PART_2Nx2N]

# Inter part modes: rectangular PUs enabled for sizes ≥ 16
# For 8×8 CUs, NxN (four 4×4 PUs) is also allowed
_INTER_PART_MODES_LARGE = [
    PartMode.PART_2Nx2N,
    PartMode.PART_2NxN,
    PartMode.PART_Nx2N,
]
_INTER_PART_MODES_8x8 = [
    PartMode.PART_2Nx2N,
    PartMode.PART_NxN,
]
_INTER_PART_MODES_4x4 = [PartMode.PART_2Nx2N]  # only 2Nx2N at minimum size

# Default intra reference padding value (mid-grey for 8-bit)
_REF_DEFAULT: int = 128


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_ctu(
    root:          QuadNode,
    frame_luma:    np.ndarray,
    ref_frame:     np.ndarray | None,
    qp:            int = 28,
    slice_type:    Literal["I", "P", "B"] = SLICE_P,
    max_depth:     int = MAX_CU_DEPTH,
    min_size:      int = MIN_CU_SIZE,
    search_range:  int = 32,
) -> float:
    """
    Recursively decide the CU split structure for a CTU subtree (in-place).

    For each node, evaluates:
        J_leaf  = RD cost of encoding the node as a single CU
        J_split = sum of children RD costs after recursive splitting

    Assigns state (SPLIT or LEAF), pred_mode, part_mode, rd_cost,
    intra_mode / mvx / mvy to every node in the tree.

    Parameters
    ----------
    root : QuadNode
        The CTU root node (or any subtree root). Modified in-place.
    frame_luma : np.ndarray
        Full luma frame, shape (H, W), dtype uint8. Used to extract
        the original pixel block and reference samples.
    ref_frame : np.ndarray or None
        Full luma reference frame for inter prediction. None forces intra.
    qp : int
        Base quantisation parameter [0, 51].
    slice_type : {"I", "P", "B"}
        Determines which prediction modes are available.
    max_depth : int
        Maximum split depth (default 4 for 64→4 CTU).
    min_size : int
        Minimum CU size in pixels (default 4).
    search_range : int
        ME integer-pel search radius (default 32).

    Returns
    -------
    float — total RD cost of the final partition
    """
    _validate_frame(frame_luma, root)

    def _recurse(node: QuadNode) -> float:
        # ── CTU root (size=64): always split, never evaluate as leaf ──────
        # mode_decision supports CU sizes {4,8,16,32} only.
        # A 64×64 block must be split into four 32×32 CUs first.
        if node.size > 32:
            if not node.can_split:
                raise ValueError(f"Cannot split oversized node {node}.")
            children = node.split()
            j_split = sum(_recurse(c) for c in children)
            node.rd_cost = j_split
            return j_split

        # ── Forced leaf at minimum size or max depth ───────────────────
        if node.size <= min_size or node.depth >= max_depth:
            return _encode_leaf(
                node, frame_luma, ref_frame, qp, slice_type, search_range
            )

        # ── Evaluate as leaf (no split) ────────────────────────────────
        j_leaf = _encode_leaf(
            node, frame_luma, ref_frame, qp, slice_type, search_range
        )

        # ── Evaluate as split (recurse into children) ──────────────────
        # Temporarily split to evaluate children
        saved_state    = node.state
        saved_children = list(node.children)

        # Reset to UNSPLIT so .split() succeeds
        node.state    = NodeState.UNSPLIT
        node.children = []
        children      = node.split()

        j_split = 0.0
        child_snapshots = []
        for child in children:
            j_child = _recurse(child)
            j_split += j_child
            child_snapshots.append(_snapshot(child))

        # ── Pick winner ────────────────────────────────────────────────
        if j_split < j_leaf:
            # Keep the split — restore child decisions
            node.rd_cost = j_split
            for child, snap in zip(children, child_snapshots):
                _restore(child, snap)
        else:
            # Revert split — restore leaf state
            node.state    = saved_state
            node.children = saved_children
            # Re-apply leaf encoding (already stored in node fields)
            _encode_leaf(
                node, frame_luma, ref_frame, qp, slice_type, search_range
            )
            node.rd_cost = j_leaf

        return node.rd_cost

    total = _recurse(root)
    return total


def evaluate_node(
    node:       QuadNode,
    frame_luma: np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    ref_frame:  np.ndarray | None,
    qp:         int,
    slice_type: Literal["I", "P", "B"] = SLICE_P,
    search_range: int = 32,
) -> ModeDecision:
    """
    Evaluate the best prediction mode for a single CU node.

    Calls mode_decision.decide_mode() with the node's pixel block and
    reference data, then stores the result in the node's fields.

    Parameters
    ----------
    node       : QuadNode — the CU to evaluate (must be UNSPLIT or LEAF)
    frame_luma : np.ndarray — full luma frame
    ref_above  : (2N+1,) int16 — intra reference samples above
    ref_left   : (2N+1,) int16 — intra reference samples left
    ref_frame  : np.ndarray or None — inter reference frame
    qp         : int — quantisation parameter
    slice_type : {"I", "P", "B"}
    search_range : int — ME search radius

    Returns
    -------
    ModeDecision — best mode decision (also written into node fields)
    """
    n     = node.size
    block = _extract_block(frame_luma, node.x, node.y, n)

    dec = decide_mode(
        block, ref_above, ref_left, ref_frame,
        origin=(node.x, node.y),
        qp=qp,
        slice_type=slice_type,
        search_range=search_range,
    )

    _apply_decision(node, dec)
    return dec


def get_ref_samples(
    frame_luma: np.ndarray,
    x:          int,
    y:          int,
    size:       int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract HEVC intra reference samples for a CU at (x, y) of `size`×`size`.

    Reads `2*size + 1` samples above the block and to the left.
    Samples outside the frame boundary are padded with `_REF_DEFAULT` (128).

    Convention (matches intra_estimation.py):
        ref_above[0]       = top-left corner
        ref_above[1:size+1] = N samples directly above
        ref_above[size+1:]  = N additional samples (top-right extension)

        ref_left[0]         = same top-left corner
        ref_left[1:size+1]  = N samples directly to the left
        ref_left[size+1:]   = N additional samples (bottom-left extension)

    Parameters
    ----------
    frame_luma : np.ndarray — full luma frame, shape (H, W), uint8
    x, y       : int        — top-left pixel of the CU
    size       : int        — CU width = height in pixels

    Returns
    -------
    (ref_above, ref_left) : tuple of (2*size+1,) int16 arrays
    """
    H, W      = frame_luma.shape
    ref_len   = 2 * size + 1
    ref_above = np.full(ref_len, _REF_DEFAULT, dtype=np.int16)
    ref_left  = np.full(ref_len, _REF_DEFAULT, dtype=np.int16)

    # Top-left corner
    if x > 0 and y > 0:
        ref_above[0] = int(frame_luma[y-1, x-1])
        ref_left[0]  = int(frame_luma[y-1, x-1])

    # Above row: ref_above[1..2*size]
    for i in range(1, ref_len):
        col = x + i - 1
        if y > 0 and 0 <= col < W:
            ref_above[i] = int(frame_luma[y-1, col])
        # else: keep default

    # Left column: ref_left[1..2*size]
    for i in range(1, ref_len):
        row = y + i - 1
        if x > 0 and 0 <= row < H:
            ref_left[i] = int(frame_luma[row, x-1])
        # else: keep default

    return ref_above, ref_left


# ---------------------------------------------------------------------------
# Partition mode evaluation
# ---------------------------------------------------------------------------

def _eligible_part_modes(
    size:       int,
    pred_mode:  str,
) -> list[PartMode]:
    """
    Return the list of eligible PU partition modes for a given CU size
    and prediction mode.

    HEVC rules:
        Intra: always 2Nx2N only
        Inter 4×4: 2Nx2N only
        Inter 8×8: 2Nx2N, NxN
        Inter ≥16: 2Nx2N, 2NxN, Nx2N
    """
    if pred_mode == MODE_INTRA:
        return _INTRA_PART_MODES
    if size == 4:
        return _INTER_PART_MODES_4x4
    if size == 8:
        return _INTER_PART_MODES_8x8
    return _INTER_PART_MODES_LARGE


def _evaluate_part_mode(
    node:       QuadNode,
    frame_luma: np.ndarray,
    ref_above:  np.ndarray,
    ref_left:   np.ndarray,
    ref_frame:  np.ndarray | None,
    part_mode:  PartMode,
    qp:         int,
    slice_type: str,
    search_range: int,
) -> tuple[float, list[ModeDecision]]:
    """
    Evaluate the RD cost of a given partition mode for a CU.

    For 2Nx2N: one mode_decision call on the full block.
    For 2NxN:  two calls on top and bottom halves.
    For Nx2N:  two calls on left and right halves.
    For NxN:   four calls on four equal quadrants.

    Returns (total_j, list_of_decisions).
    """
    n = node.size

    if part_mode == PartMode.PART_2Nx2N:
        block = _extract_block(frame_luma, node.x, node.y, n)
        dec   = decide_mode(
            block, ref_above, ref_left, ref_frame,
            origin=(node.x, node.y), qp=qp,
            slice_type=slice_type, search_range=search_range,
        )
        return dec.rd_cost, [dec]

    elif part_mode == PartMode.PART_2NxN:
        # Two horizontal halves: top (x, y, n, n//2), bottom (x, y+n//2, n, n//2)
        hn = n // 2
        total_j = 0.0; decs = []
        for dy in [0, hn]:
            sub_block = _extract_block(frame_luma, node.x, node.y + dy, n)
            # Use same intra refs — simplified (full-CU refs)
            ra, rl = get_ref_samples(frame_luma, node.x, node.y + dy, n)
            d = decide_mode(sub_block, ra, rl, ref_frame,
                            origin=(node.x, node.y + dy), qp=qp,
                            slice_type=slice_type, search_range=search_range)
            total_j += d.rd_cost; decs.append(d)
        return total_j, decs

    elif part_mode == PartMode.PART_Nx2N:
        # Two vertical halves: left (x, y, n//2, n), right (x+n//2, y, n//2, n)
        hn = n // 2
        total_j = 0.0; decs = []
        for dx in [0, hn]:
            sub_block = _extract_block(frame_luma, node.x + dx, node.y, n)
            ra, rl = get_ref_samples(frame_luma, node.x + dx, node.y, n)
            d = decide_mode(sub_block, ra, rl, ref_frame,
                            origin=(node.x + dx, node.y), qp=qp,
                            slice_type=slice_type, search_range=search_range)
            total_j += d.rd_cost; decs.append(d)
        return total_j, decs

    elif part_mode == PartMode.PART_NxN:
        # Four quadrants, each n//2 × n//2
        hn = n // 2
        total_j = 0.0; decs = []
        for dy in [0, hn]:
            for dx in [0, hn]:
                sub_block = _extract_block(frame_luma, node.x+dx, node.y+dy, hn)
                ra, rl = get_ref_samples(frame_luma, node.x+dx, node.y+dy, hn)
                d = decide_mode(sub_block, ra, rl, ref_frame,
                                origin=(node.x+dx, node.y+dy), qp=qp,
                                slice_type=slice_type, search_range=search_range)
                total_j += d.rd_cost; decs.append(d)
        return total_j, decs

    else:
        raise ValueError(f"Unknown partition mode: {part_mode}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _encode_leaf(
    node:         QuadNode,
    frame_luma:   np.ndarray,
    ref_frame:    np.ndarray | None,
    qp:           int,
    slice_type:   str,
    search_range: int,
) -> float:
    """
    Evaluate all eligible partition modes for `node` as a leaf CU,
    pick the best by RD cost, and write results into node fields.

    Returns the winning RD cost.
    """
    ra, rl = get_ref_samples(frame_luma, node.x, node.y, node.size)

    # First: find the best prediction mode (intra/inter/skip)
    dec = decide_mode(
        _extract_block(frame_luma, node.x, node.y, node.size),
        ra, rl, ref_frame,
        origin=(node.x, node.y),
        qp=qp, slice_type=slice_type,
        search_range=search_range,
    )

    # Then: for inter nodes, try rectangular partition modes
    best_j    = dec.rd_cost
    best_dec  = dec
    best_part = PartMode.PART_2Nx2N

    if dec.mode in (MODE_INTER, MODE_SKIP) and slice_type != SLICE_I:
        for pm in _eligible_part_modes(node.size, dec.mode):
            if pm == PartMode.PART_2Nx2N:
                continue   # already evaluated above
            j_pm, _ = _evaluate_part_mode(
                node, frame_luma, ra, rl, ref_frame,
                pm, qp, slice_type, search_range,
            )
            if j_pm < best_j:
                best_j    = j_pm
                best_part = pm

    # Commit best decision to node
    _apply_decision(node, best_dec)
    node.part_mode = best_part
    node.rd_cost   = best_j
    node.mark_leaf()
    return best_j


def _apply_decision(node: QuadNode, dec: ModeDecision) -> None:
    """Write a ModeDecision result into a QuadNode's fields."""
    node.qp      = dec.qp
    node.rd_cost = dec.rd_cost

    if dec.mode == MODE_INTRA:
        node.pred_mode  = PredMode.INTRA
        node.skip_flag  = False
        node.intra_mode = dec.intra_mode or 0
        if dec.intra_result:
            node.pred     = dec.intra_result.pred
            node.residual = dec.intra_result.residual
            node.coeffs   = dec.intra_result.dct_coeffs
            node.levels   = dec.intra_result.quant_levels

    elif dec.mode == MODE_INTER:
        node.pred_mode = PredMode.INTER
        node.skip_flag = False
        if dec.mv:
            node.mvx     = dec.mv.mvx
            node.mvy     = dec.mv.mvy
            node.ref_idx = dec.mv.ref_idx
        if dec.inter_result:
            node.pred     = dec.inter_result.pred
            node.residual = dec.inter_result.residual
            node.coeffs   = dec.inter_result.dct_coeffs
            node.levels   = dec.inter_result.quant_levels

    elif dec.mode == MODE_SKIP:
        node.pred_mode = PredMode.SKIP
        node.skip_flag = True
        if dec.mv:
            node.mvx     = dec.mv.mvx
            node.mvy     = dec.mv.mvy
            node.ref_idx = dec.mv.ref_idx
        if dec.inter_result:
            node.pred     = dec.inter_result.pred
            node.residual = dec.inter_result.residual
            node.levels   = dec.inter_result.quant_levels


def _extract_block(
    frame: np.ndarray,
    x:     int,
    y:     int,
    size:  int,
) -> np.ndarray:
    """
    Extract a square block from the frame, clamping at boundaries.
    Returns a (size, size) uint8 array.
    """
    H, W = frame.shape
    y1, y2 = max(0, y), min(H, y + size)
    x1, x2 = max(0, x), min(W, x + size)
    patch = np.full((size, size), _REF_DEFAULT, dtype=np.uint8)
    ph, pw = y2 - y1, x2 - x1
    if ph > 0 and pw > 0:
        patch[:ph, :pw] = frame[y1:y2, x1:x2]
    return patch


def _validate_frame(frame: np.ndarray, root: QuadNode) -> None:
    if frame.ndim != 2:
        raise ValueError(f"frame_luma must be 2-D, got {frame.ndim}-D.")
    H, W = frame.shape
    if root.x + root.size > W or root.y + root.size > H:
        raise ValueError(
            f"CTU at ({root.x},{root.y}) size={root.size} exceeds "
            f"frame {W}×{H}."
        )


# ── Snapshot / restore helpers for RDO backtracking ─────────────────────────

def _snapshot(node: QuadNode) -> dict:
    """Capture the mutable fields of a node for later restoration."""
    return {
        "state":      node.state,
        "children":   list(node.children),
        "pred_mode":  node.pred_mode,
        "part_mode":  node.part_mode,
        "qp":         node.qp,
        "rd_cost":    node.rd_cost,
        "skip_flag":  node.skip_flag,
        "mvx":        node.mvx,
        "mvy":        node.mvy,
        "ref_idx":    node.ref_idx,
        "intra_mode": node.intra_mode,
        "pred":       node.pred,
        "residual":   node.residual,
        "coeffs":     node.coeffs,
        "levels":     node.levels,
        "recon":      node.recon,
    }


def _restore(node: QuadNode, snap: dict) -> None:
    """Restore a node's mutable fields from a snapshot."""
    for k, v in snap.items():
        setattr(node, k, v)