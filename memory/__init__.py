"""
memory/__init__.py
The Memory Management Layer for the HEVC Golden Model.

This package encapsulates the Decoded Picture Buffer (DPB) for storing
reconstructed frames and the Reference Manager for building Reference Picture
Lists (RPL0 / RPL1) used in Inter prediction and bitstream signalling.
"""

# ---------------------------------------------------------------------------
# 1. Decoded Picture Buffer (Physical Storage)
# ---------------------------------------------------------------------------
from .decoded_picture_buffer import (
    DecodedPictureBuffer,
    PictureBuffer,
    FrameType,
    RefStatus,
    make_picture_buffer,
    make_grey_frame,
)

# ---------------------------------------------------------------------------
# 2. Reference Picture Management (Logical Lists & RPS)
# ---------------------------------------------------------------------------
from .reference_manager import (
    ReferenceManager,
    ReferenceList,
    ReferencePictureSet,
    ReferenceEntry,
)

# ---------------------------------------------------------------------------
# Public API Export List
# ---------------------------------------------------------------------------
__all__ = [
    # DPB Data Structures
    "DecodedPictureBuffer",
    "PictureBuffer",
    "FrameType",
    "RefStatus",
    "make_picture_buffer",
    "make_grey_frame",

    # Reference Management & RPL
    "ReferenceManager",
    "ReferenceList",
    "ReferencePictureSet",
    "ReferenceEntry",
]