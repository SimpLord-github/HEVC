"""
prediction/__init__.py
The Prediction Layer for the HEVC Golden Model.

This package encapsulates both Intra (spatial) and Inter (temporal) prediction,
as well as the Rate-Distortion Optimization (RDO) logic to choose between them.
"""

# ---------------------------------------------------------------------------
# 1. Intra Prediction (Spatial)
# ---------------------------------------------------------------------------
from prediction.intra_estimation import (
    estimate_intra_mode,
    BestMode,
    PLANAR_MODE,
    DC_MODE,
)

from prediction.intra_prediction import (
    predict_intra_luma,
    predict_intra_chroma,
    generate_residual,
    full_intra_pipeline,
    IntraResult,
)

# ---------------------------------------------------------------------------
# 2. Inter Prediction (Temporal)
# ---------------------------------------------------------------------------
from prediction.motion_estimation import (
    estimate_motion,
    MotionVector,
    ALGO_HEX,
    ALGO_FULL,
    ALGO_UMH,
)

from prediction.motion_compensation import (
    compensate_luma,
    compensate_chroma,
    compensate_bi,
    generate_inter_residual,
    full_inter_pipeline,
    InterResult,
)

# ---------------------------------------------------------------------------
# 3. Rate-Distortion Optimization (RDO)
# ---------------------------------------------------------------------------
from prediction.mode_decision import (
    decide_mode,
    ModeDecision,
    MODE_INTRA,
    MODE_INTER,
    MODE_SKIP,
)

# ---------------------------------------------------------------------------
# Public API Export List
# ---------------------------------------------------------------------------
__all__ = [
    # Intra API
    "estimate_intra_mode",
    "BestMode",
    "PLANAR_MODE",
    "DC_MODE",
    "predict_intra_luma",
    "predict_intra_chroma",
    "generate_residual",
    "full_intra_pipeline",
    "IntraResult",

    # Inter API
    "estimate_motion",
    "MotionVector",
    "ALGO_HEX",
    "ALGO_FULL",
    "ALGO_UMH",
    "compensate_luma",
    "compensate_chroma",
    "compensate_bi",
    "generate_inter_residual",
    "full_inter_pipeline",
    "InterResult",

    # RDO Mode Decision API
    "decide_mode",
    "ModeDecision",
    "MODE_INTRA",
    "MODE_INTER",
    "MODE_SKIP",
]