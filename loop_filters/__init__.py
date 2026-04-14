"""
loop_filter/__init__.py
The In-Loop Filtering Layer for the HEVC Golden Model.

This package encapsulates the post-reconstruction filters applied to 
frames before they are stored in the Decoded Picture Buffer (DPB).
It includes the Deblocking Filter (DBF) for edge smoothing and 
the Sample Adaptive Offset (SAO) for banding/ringing artifact reduction.
"""

# ---------------------------------------------------------------------------
# 1. Deblocking Filter (DBF)
# ---------------------------------------------------------------------------
from .deblocking import (
    deblock_frame,
    compute_boundary_strength,
    compute_beta,
    compute_tc,
)

# ---------------------------------------------------------------------------
# 2. Sample Adaptive Offset (SAO)
# ---------------------------------------------------------------------------
from .sao import (
    sao_filter_frame,
    estimate_sao_params,
    estimate_sao_frame,
    SAOParams,
    SAOType,
    EODirection,
)

# ---------------------------------------------------------------------------
# Public API Export List
# ---------------------------------------------------------------------------
__all__ = [
    # Deblocking API
    "deblock_frame",
    "compute_boundary_strength",
    "compute_beta",
    "compute_tc",

    # SAO API
    "sao_filter_frame",
    "estimate_sao_params",
    "estimate_sao_frame",
    "SAOParams",
    "SAOType",
    "EODirection",
]