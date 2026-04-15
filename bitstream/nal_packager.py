"""
nal_packager.py — HEVC NAL Unit Packager (Golden Model)
See slice_header_and_nal.py for full documentation.
Re-exports all public symbols from the combined module.
"""
from bitstream.slice_header_and_nal import (
    NalPackager, _build_nal, _apply_emulation_prevention,
    START_CODE_3, START_CODE_4,
)
