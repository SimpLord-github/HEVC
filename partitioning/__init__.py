"""
partitioning/__init__.py
The Spatial Partitioning Layer for the HEVC Golden Model.

This package manages the hierarchical subdivision of frames:
CTU (Coding Tree Unit) -> CU (Coding Unit) -> PU / TU (Prediction / Transform Units).
It provides the core Quad-Tree data structures and the RDO-driven split logic.
"""

# ---------------------------------------------------------------------------
# 1. Data Structures (The Skeleton)
# ---------------------------------------------------------------------------
from .quad_tree import (
    QuadNode,
    QuadTree,
    NodeState,
    PredMode,
    PartMode,
    build_full_tree,
    build_flat_tree,
    build_uniform_tree,
    CTU_SIZE,
    MIN_CU_SIZE,
    MAX_CU_DEPTH,
)

# ---------------------------------------------------------------------------
# 2. Coding Unit (CU) Split Decision (The Brain for Prediction)
# ---------------------------------------------------------------------------
from .cu_split import (
    split_ctu,
    evaluate_node,
    get_ref_samples,
)

# ---------------------------------------------------------------------------
# 3. Transform Unit (TU) Split Decision (The Brain for Residuals)
# ---------------------------------------------------------------------------
from .tu_split import (
    split_tu,
    TUResult,
    TULeafResult,
    reconstruct_cu,
)

# ---------------------------------------------------------------------------
# Public API Export List
# ---------------------------------------------------------------------------
__all__ = [
    # Data Structures & Enums
    "QuadNode",
    "QuadTree",
    "NodeState",
    "PredMode",
    "PartMode",
    "CTU_SIZE",
    "MIN_CU_SIZE",
    "MAX_CU_DEPTH",
    
    # Tree Builders
    "build_full_tree",
    "build_flat_tree",
    "build_uniform_tree",

    # CU API
    "split_ctu",
    "evaluate_node",
    "get_ref_samples",

    # TU API
    "split_tu",
    "TUResult",
    "TULeafResult",
    "reconstruct_cu",
]