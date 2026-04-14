"""
quad_tree.py — HEVC Quad-Tree Data Structures
Pure data structures for the CTU → CU → PU / TU recursive partition tree.
No split decisions are made here — only nodes, trees, and traversal.

Pipeline position
-----------------
    ctu_splitter  →  [quad_tree.py]  →  cu_split.py / tu_split.py
    (raw CTU)         (tree nodes)       (split decisions)

HEVC partition hierarchy (ISO/IEC 23008-2 §7.3.8)
---------------------------------------------------
    CTU  (Coding Tree Unit)   — 64×64 luma samples, top-level block
     └─ CU  (Coding Unit)     — power-of-2 square, split or leaf
         ├─ PU  (Prediction Unit) — mode + prediction signal (at leaf CU)
         └─ TU  (Transform Unit)  — residual transform block (own quad-tree)

Two independent quad-trees per CTU:
    CU quad-tree  — controls how the 64×64 is split into CUs
    TU quad-tree  — within each leaf CU, controls how residual is split

Coordinate convention
----------------------
    All coordinates are in luma PIXEL units.
    x, y = top-left corner of the block within the frame.
    size  = width = height (always square).

Depth convention
-----------------
    depth = 0  → CTU (64×64, or top-level CU)
    depth = 1  → 32×32 children
    depth = 2  → 16×16 children
    depth = 3  →  8×8  children
    depth = 4  →  4×4  children (minimum CU size)

Node states
-----------
    UNSPLIT — leaf node, not yet evaluated by the split decision logic
    SPLIT   — internal node, has four children (NW, NE, SW, SE)
    LEAF    — final leaf: will be encoded as a CU/TU at this depth

Public API
----------
    QuadNode          — single node in the quad-tree
    QuadTree          — tree of QuadNodes rooted at the CTU
    build_full_tree() — build a fully-split tree down to min_size
    build_flat_tree() — build a single-node tree (no split)
    build_uniform_tree() — split uniformly to a given CU size
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Iterator, Callable
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CTU_SIZE:     int = 64   # largest coding unit
MIN_CU_SIZE:  int = 4    # smallest CU / TU
MIN_TU_SIZE:  int = 4
MAX_CU_DEPTH: int = 4    # log2(CTU_SIZE / MIN_CU_SIZE)

VALID_CU_SIZES = (64, 32, 16, 8, 4)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NodeState(Enum):
    UNSPLIT = auto()   # not yet decided
    SPLIT   = auto()   # has four children
    LEAF    = auto()   # terminal — encode at this depth


class PredMode(Enum):
    """Prediction mode assigned to a leaf CU by mode_decision.py."""
    NONE  = auto()   # not yet decided
    INTRA = auto()
    INTER = auto()
    SKIP  = auto()


class PartMode(Enum):
    """
    Prediction unit partition mode within a leaf CU.
    HEVC §7.3.8.5 — only 2Nx2N is used for intra.
    Inter supports rectangular PUs.
    """
    PART_2Nx2N = "2Nx2N"   # single PU covering the whole CU (intra + inter)
    PART_2NxN  = "2NxN"    # two PUs stacked horizontally
    PART_Nx2N  = "Nx2N"    # two PUs side by side
    PART_NxN   = "NxN"     # four equal PUs (only for 8×8 CUs)


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------

@dataclass
class QuadNode:
    """
    One node in the CTU coding quad-tree.

    A node represents a square block at (x, y) of `size`×`size` luma pixels.
    It is either a LEAF (encoded at this size) or SPLIT into four children.

    Children are ordered: NW, NE, SW, SE (raster order).
    The relationship is:
        NW: (x,           y          ), size//2
        NE: (x + size//2, y          ), size//2
        SW: (x,           y + size//2), size//2
        SE: (x + size//2, y + size//2), size//2

    Attributes assigned by cu_split.py / mode_decision.py
    -------------------------------------------------------
    state     — UNSPLIT / SPLIT / LEAF
    pred_mode — INTRA / INTER / SKIP (only meaningful at LEAF)
    part_mode — PU partition within this CU (only at LEAF)
    qp        — effective QP for this node (may differ from frame QP with QP delta)
    rd_cost   — RD cost of the best mode decision for this node
    skip_flag — True if this node is encoded as SKIP (zero residual)

    Attributes assigned by the full pipeline
    -----------------------------------------
    pred      — prediction pixel block, shape (size, size), uint8
    residual  — residual block, shape (size, size), int16
    coeffs    — DCT coefficients, shape (size, size), int32
    levels    — quantised levels, shape (size, size), int32
    recon     — reconstructed block after IDCT+pred, shape (size, size), uint8
    """
    # Geometry — immutable after creation
    x:    int         # top-left column in luma frame pixels
    y:    int         # top-left row in luma frame pixels
    size: int         # block width = height
    depth: int = 0    # 0 = CTU level

    # Tree structure
    state:    NodeState      = field(default=NodeState.UNSPLIT)
    children: list[QuadNode] = field(default_factory=list)   # len 0 or 4

    # Mode decision fields (set by cu_split / mode_decision)
    pred_mode: PredMode = field(default=PredMode.NONE)
    part_mode: PartMode = field(default=PartMode.PART_2Nx2N)
    qp:        int      = field(default=28)
    rd_cost:   float    = field(default=float("inf"))
    skip_flag: bool     = field(default=False)

    # Motion vector (set by motion_estimation, for INTER/SKIP nodes)
    mvx: int = field(default=0)   # qpel units
    mvy: int = field(default=0)   # qpel units
    ref_idx: int = field(default=0)

    # Intra mode (set by intra_estimation, for INTRA nodes)
    intra_mode: int = field(default=0)

    # Pixel data (set by the encoding pipeline, optional)
    pred:     np.ndarray | None = field(default=None, repr=False)
    residual: np.ndarray | None = field(default=None, repr=False)
    coeffs:   np.ndarray | None = field(default=None, repr=False)
    levels:   np.ndarray | None = field(default=None, repr=False)
    recon:    np.ndarray | None = field(default=None, repr=False)

    # ── Convenience properties ──────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        return self.state == NodeState.LEAF

    @property
    def is_split(self) -> bool:
        return self.state == NodeState.SPLIT

    @property
    def is_unsplit(self) -> bool:
        return self.state == NodeState.UNSPLIT

    @property
    def area(self) -> int:
        return self.size * self.size

    @property
    def can_split(self) -> bool:
        """True if splitting would produce children ≥ MIN_CU_SIZE."""
        return self.size > MIN_CU_SIZE

    @property
    def child_size(self) -> int:
        return self.size // 2

    @property
    def origin(self) -> tuple[int, int]:
        """(x, y) top-left corner."""
        return (self.x, self.y)

    @property
    def nw(self) -> QuadNode | None:
        return self.children[0] if len(self.children) == 4 else None

    @property
    def ne(self) -> QuadNode | None:
        return self.children[1] if len(self.children) == 4 else None

    @property
    def sw(self) -> QuadNode | None:
        return self.children[2] if len(self.children) == 4 else None

    @property
    def se(self) -> QuadNode | None:
        return self.children[3] if len(self.children) == 4 else None

    # ── Tree mutation ───────────────────────────────────────────────────

    def split(self) -> list[QuadNode]:
        """
        Split this node into four equal children (NW, NE, SW, SE).

        Sets state to SPLIT and populates self.children.
        Raises ValueError if the node cannot be split (already at min size)
        or is already split/leaf.

        Returns
        -------
        list[QuadNode] — the four new children in raster order
        """
        if not self.can_split:
            raise ValueError(
                f"Cannot split node at depth {self.depth}: "
                f"size {self.size} is already at minimum ({MIN_CU_SIZE})."
            )
        if self.state != NodeState.UNSPLIT:
            raise ValueError(
                f"Node at ({self.x},{self.y}) size={self.size} is already "
                f"{self.state.name} and cannot be split again."
            )

        hs = self.size // 2   # half size
        self.children = [
            QuadNode(self.x,      self.y,      hs, self.depth + 1, qp=self.qp),  # NW
            QuadNode(self.x + hs, self.y,      hs, self.depth + 1, qp=self.qp),  # NE
            QuadNode(self.x,      self.y + hs, hs, self.depth + 1, qp=self.qp),  # SW
            QuadNode(self.x + hs, self.y + hs, hs, self.depth + 1, qp=self.qp),  # SE
        ]
        self.state = NodeState.SPLIT
        return self.children

    def mark_leaf(self) -> None:
        """Mark this node as a terminal leaf CU."""
        if self.state == NodeState.SPLIT:
            raise ValueError("Cannot mark a SPLIT node as LEAF.")
        self.state = NodeState.LEAF

    def mark_unsplit(self) -> None:
        """Reset to UNSPLIT and clear all children."""
        self.children.clear()
        self.state = NodeState.UNSPLIT

    # ── Traversal ───────────────────────────────────────────────────────

    def leaves(self) -> Iterator[QuadNode]:
        """
        Depth-first iterator over all LEAF (and UNSPLIT) terminal nodes.
        """
        if self.state != NodeState.SPLIT:
            yield self
        else:
            for child in self.children:
                yield from child.leaves()

    def all_nodes(self) -> Iterator[QuadNode]:
        """Depth-first iterator over every node (internal + leaves)."""
        yield self
        for child in self.children:
            yield from child.all_nodes()

    def nodes_at_depth(self, target_depth: int) -> Iterator[QuadNode]:
        """Yield all nodes exactly at `target_depth`."""
        if self.depth == target_depth:
            yield self
        else:
            for child in self.children:
                yield from child.nodes_at_depth(target_depth)

    def apply(self, fn: Callable[[QuadNode], None]) -> None:
        """Apply a function to every node depth-first."""
        fn(self)
        for child in self.children:
            child.apply(fn)

    # ── Statistics ──────────────────────────────────────────────────────

    def count_leaves(self) -> int:
        return sum(1 for _ in self.leaves())

    def count_nodes(self) -> int:
        return sum(1 for _ in self.all_nodes())

    def max_depth(self) -> int:
        if not self.children:
            return self.depth
        return max(c.max_depth() for c in self.children)

    # ── Representation ──────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = self.pred_mode.name if self.pred_mode != PredMode.NONE else "?"
        return (f"QuadNode(x={self.x}, y={self.y}, size={self.size}, "
                f"depth={self.depth}, state={self.state.name}, mode={mode})")

    def pretty(self, indent: int = 0) -> str:
        """Multi-line tree visualisation for debugging."""
        pad  = "  " * indent
        line = f"{pad}[{self.state.name}] {self.size}×{self.size} @ ({self.x},{self.y})"
        if self.pred_mode != PredMode.NONE:
            line += f"  mode={self.pred_mode.name}"
        if self.state == NodeState.SPLIT:
            child_lines = "\n".join(c.pretty(indent + 1) for c in self.children)
            return line + "\n" + child_lines
        return line


# ---------------------------------------------------------------------------
# QuadTree — wrapper around the root node
# ---------------------------------------------------------------------------

@dataclass
class QuadTree:
    """
    Quad-tree representing the complete partition of one CTU.

    Attributes
    ----------
    root    — root QuadNode (the CTU)
    frame_w — luma frame width in pixels
    frame_h — luma frame height in pixels
    """
    root:    QuadNode
    frame_w: int = 0
    frame_h: int = 0

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_ctu(
        cls,
        ctu_x:    int,
        ctu_y:    int,
        ctu_size: int = CTU_SIZE,
        qp:       int = 28,
        frame_w:  int = 0,
        frame_h:  int = 0,
    ) -> QuadTree:
        """Create a flat (unsplit) tree for one CTU."""
        root = QuadNode(x=ctu_x, y=ctu_y, size=ctu_size, depth=0, qp=qp)
        return cls(root=root, frame_w=frame_w, frame_h=frame_h)

    # ── Traversal shortcuts ──────────────────────────────────────────────

    def leaves(self) -> Iterator[QuadNode]:
        return self.root.leaves()

    def all_nodes(self) -> Iterator[QuadNode]:
        return self.root.all_nodes()

    def nodes_at_depth(self, depth: int) -> Iterator[QuadNode]:
        return self.root.nodes_at_depth(depth)

    def apply(self, fn: Callable[[QuadNode], None]) -> None:
        self.root.apply(fn)

    # ── Statistics ───────────────────────────────────────────────────────

    def count_leaves(self) -> int:
        return self.root.count_leaves()

    def count_nodes(self) -> int:
        return self.root.count_nodes()

    def max_depth(self) -> int:
        return self.root.max_depth()

    def leaf_sizes(self) -> dict[int, int]:
        """Return {size: count} histogram of leaf sizes."""
        hist: dict[int, int] = {}
        for node in self.leaves():
            hist[node.size] = hist.get(node.size, 0) + 1
        return hist

    def total_area(self) -> int:
        """Sum of areas of all leaf nodes — must equal root area."""
        return sum(n.area for n in self.leaves())

    # ── Validation ───────────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """
        Run structural consistency checks. Returns a list of error strings.
        Empty list = valid tree.

        Checks:
            - Split nodes have exactly 4 children with correct geometry
            - Children depths = parent depth + 1
            - No node smaller than MIN_CU_SIZE
            - Total leaf area equals root area
        """
        errors: list[str] = []

        def _check(node: QuadNode) -> None:
            if node.size < MIN_CU_SIZE:
                errors.append(f"Node {node} size < {MIN_CU_SIZE}.")

            if node.state == NodeState.SPLIT:
                if len(node.children) != 4:
                    errors.append(
                        f"SPLIT node {node} has {len(node.children)} children."
                    )
                    return
                hs = node.size // 2
                expected = [
                    (node.x,      node.y,      hs),
                    (node.x + hs, node.y,      hs),
                    (node.x,      node.y + hs, hs),
                    (node.x + hs, node.y + hs, hs),
                ]
                for child, (ex, ey, es) in zip(node.children, expected):
                    if (child.x, child.y, child.size) != (ex, ey, es):
                        errors.append(
                            f"Child geometry wrong: got ({child.x},{child.y},{child.size}), "
                            f"expected ({ex},{ey},{es})."
                        )
                    if child.depth != node.depth + 1:
                        errors.append(
                            f"Child depth {child.depth} ≠ parent depth {node.depth} + 1."
                        )
                for child in node.children:
                    _check(child)
            else:
                if node.children:
                    errors.append(
                        f"Non-SPLIT node {node} unexpectedly has children."
                    )

        _check(self.root)

        # Area conservation — always check, even when geometry errors exist
        if self.total_area() != self.root.area:
            errors.append(
                f"Area mismatch: leaf total {self.total_area()} ≠ "
                f"root area {self.root.area}."
            )

        return errors

    # ── Representation ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (f"QuadTree(root=({self.root.x},{self.root.y}), "
                f"size={self.root.size}, leaves={self.count_leaves()}, "
                f"max_depth={self.max_depth()})")

    def pretty(self) -> str:
        return self.root.pretty()


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def build_full_tree(
    ctu_x:    int,
    ctu_y:    int,
    min_size: int = MIN_CU_SIZE,
    ctu_size: int = CTU_SIZE,
    qp:       int = 28,
    frame_w:  int = 0,
    frame_h:  int = 0,
) -> QuadTree:
    """
    Build a maximally-split quad-tree down to `min_size`.

    All nodes that can split (size > min_size) are split.
    Every leaf ends up at `min_size` × `min_size`.
    """
    tree = QuadTree.from_ctu(ctu_x, ctu_y, ctu_size, qp, frame_w, frame_h)

    def _recurse(node: QuadNode) -> None:
        if node.size > min_size and node.can_split:
            for child in node.split():
                _recurse(child)
        else:
            node.mark_leaf()

    _recurse(tree.root)
    return tree


def build_flat_tree(
    ctu_x:    int,
    ctu_y:    int,
    ctu_size: int = CTU_SIZE,
    qp:       int = 28,
    frame_w:  int = 0,
    frame_h:  int = 0,
) -> QuadTree:
    """
    Build a single-node (no-split) tree. The CTU is one big LEAF CU.
    """
    tree = QuadTree.from_ctu(ctu_x, ctu_y, ctu_size, qp, frame_w, frame_h)
    tree.root.mark_leaf()
    return tree


def build_uniform_tree(
    ctu_x:    int,
    ctu_y:    int,
    cu_size:  int,
    ctu_size: int = CTU_SIZE,
    qp:       int = 28,
    frame_w:  int = 0,
    frame_h:  int = 0,
) -> QuadTree:
    """
    Build a uniformly-split tree where every leaf is `cu_size` × `cu_size`.

    Parameters
    ----------
    cu_size : int — target leaf size, must be in VALID_CU_SIZES and ≤ ctu_size
    """
    if cu_size not in VALID_CU_SIZES or cu_size > ctu_size:
        raise ValueError(
            f"cu_size must be in {VALID_CU_SIZES} and ≤ {ctu_size}. Got {cu_size}."
        )
    return build_full_tree(ctu_x, ctu_y, min_size=cu_size,
                           ctu_size=ctu_size, qp=qp,
                           frame_w=frame_w, frame_h=frame_h)