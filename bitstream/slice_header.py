"""
slice_header.py — HEVC Slice Header Builder (Golden Model)
See slice_header_and_nal.py for full documentation.
Re-exports all public symbols from the combined module.
"""
from bitstream.slice_header_and_nal import (
    SliceHeader, build_slice_header_rbsp,
    _RBSPWriter, _build_vps_rbsp, _build_sps_rbsp, _build_pps_rbsp,
    NAL_VPS, NAL_SPS, NAL_PPS, NAL_IDR, NAL_TRAIL_R,
)
