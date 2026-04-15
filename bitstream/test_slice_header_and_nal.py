import pytest
import struct

from slice_header_and_nal import (
    _RBSPWriter,
    _apply_emulation_prevention,
    _build_nal,
    SliceHeader,
    build_slice_header_rbsp,
    NalPackager,
    START_CODE_3,
    START_CODE_4,
    NAL_VPS,
    NAL_SPS,
    NAL_PPS,
    NAL_IDR,
    NAL_TRAIL_R
)

# ===========================================================================
# 1. TEST LÕI GHI BIT (RBSP WRITER & EXP-GOLOMB)
# ===========================================================================

def test_rbsp_writer_unsigned_fixed():
    """Kiểm tra ghi các bit cố định (u)."""
    w = _RBSPWriter()
    w.u(0b1011, 4)  # Ghi 4 bit: 1011
    w.u(0b001, 3)   # Ghi 3 bit: 001
    w.flag(True)    # Ghi 1 bit: 1
    # Tổng: 8 bit -> 1011 001 1 -> 0xB3
    
    assert w.get_bytes() == bytes([0xB3]), "Lỗi: Ghi bit cố định sai!"

def test_rbsp_writer_exp_golomb_ue():
    """Kiểm tra mã hóa Exp-Golomb không dấu (ue)."""
    w = _RBSPWriter()
    w.ue(0)  # 1
    w.ue(1)  # 010
    w.ue(2)  # 011
    w.trailing_bits() # Byte alignment: 10...
    
    # Chuỗi bit: 1_010_011_1 = 10100111 = 0xA7
    assert w.get_bytes() == bytes([0xA7]), "Lỗi: Mã Exp-Golomb UE (Unsigned) sai chuẩn!"

def test_rbsp_writer_exp_golomb_se():
    """Kiểm tra mã hóa Exp-Golomb có dấu (se)."""
    w = _RBSPWriter()
    w.se(0)   # ue(0) -> 1
    w.se(1)   # ue(1) -> 010
    w.se(-1)  # ue(2) -> 011
    w.trailing_bits()
    
    # Chuỗi bit: 1_010_011_1 = 0xA7
    assert w.get_bytes() == bytes([0xA7]), "Lỗi: Mã Exp-Golomb SE (Signed) sai chuẩn!"


# ===========================================================================
# 2. TEST LÕI BẢO VỆ DỮ LIỆU (EMULATION PREVENTION)
# ===========================================================================

def test_emulation_prevention_byte_insertion():
    """
    HEVC cấm chuỗi 0x000000, 0x000001, 0x000002, 0x000003 xuất hiện trong Payload.
    Nếu có, phải chèn 0x03 vào trước byte cuối.
    """
    # Test 1: Nguy cơ Start Code (00 00 01)
    payload_1 = bytes([0x00, 0x00, 0x01, 0xFF])
    safe_1 = _apply_emulation_prevention(payload_1)
    assert list(safe_1) == [0x00, 0x00, 0x03, 0x01, 0xFF]

    # Test 2: Nguy cơ 4 byte 0 (00 00 00 00)
    payload_2 = bytes([0x00, 0x00, 0x00, 0x00])
    safe_2 = _apply_emulation_prevention(payload_2)
    assert list(safe_2) == [0x00, 0x00, 0x03, 0x00, 0x00]

    # Test 3: Chuỗi an toàn không bị biến đổi
    payload_3 = bytes([0x00, 0x11, 0x00, 0x22])
    safe_3 = _apply_emulation_prevention(payload_3)
    assert safe_3 == payload_3


# ===========================================================================
# 3. TEST ĐÓNG GÓI NAL (NAL FRAMING & START CODES)
# ===========================================================================

def test_build_nal_header_structure():
    """Đảm bảo NAL Header 2 byte được pack đúng cấu trúc bit."""
    # Thử pack một NAL_SPS (33)
    nal = _build_nal(NAL_SPS, rbsp=b'\xFF', use_4byte_start=True)
    
    # Start code 4 bytes: 00 00 00 01
    assert nal[:4] == START_CODE_4
    
    # NAL Header 2 bytes:
    # F(1)=0 | Type(6)=33 | Layer(6)=0 | TempId(3)=1
    # Byte 1: 0 100001 0 -> 0x42
    # Byte 2: 00000 001 -> 0x01
    assert nal[4:6] == bytes([0x42, 0x01]), "Lỗi: Cấu trúc NAL Header 2-byte bị sai!"


# ===========================================================================
# 4. TEST TÍCH HỢP: NAL PACKAGER
# ===========================================================================

def test_nal_packager_parameter_sets():
    """Kiểm tra Packager sinh ra đủ 3 Parameter Sets (VPS, SPS, PPS)."""
    packager = NalPackager(width=1920, height=1080)
    
    param_stream = packager.write_parameter_sets()
    
    # Phải có ít nhất 3 Start Codes (cho VPS, SPS, PPS)
    start_code_count = param_stream.count(START_CODE_4)
    assert start_code_count == 3, "Lỗi: Không sinh đủ VPS, SPS, PPS!"
    assert packager.param_sets_written is True

def test_nal_packager_idr_slice():
    """Kiểm tra đóng gói IDR Slice (Khung I đầu tiên)."""
    packager = NalPackager(width=1920, height=1080)
    
    mock_cabac = bytes([0xAA, 0xBB, 0xCC])
    slice_stream = packager.write_slice(mock_cabac, poc=0, is_idr=True, slice_type="I")
    
    # IDR Slice / Khung POC 0 phải dùng Start Code 4 bytes
    assert slice_stream.startswith(START_CODE_4)
    
    # NAL Type phải là 19 (IDR_W_RADL)
    # Byte header: 0 010011 0 -> 0x26
    assert slice_stream[4] == 0x26, "Lỗi: Slice Header không đánh dấu đúng IDR NAL Type!"

def test_nal_packager_trailing_slice():
    """Kiểm tra đóng gói Trailing Slice (Khung P/B)."""
    packager = NalPackager(width=1920, height=1080)
    
    mock_cabac = bytes([0xAA, 0xBB, 0xCC])
    # Khung P, POC = 1, trỏ về POC 0 (delta = -1)
    slice_stream = packager.write_slice(
        mock_cabac, 
        poc=1, 
        is_idr=False, 
        slice_type="P",
        num_negative=1,
        delta_poc_neg=[-1]
    )
    
    # Các khung P/B sau khung I phải dùng Start Code 3 bytes
    assert slice_stream.startswith(START_CODE_3)
    
    # NAL Type phải là 1 (TRAIL_R)
    # Byte header: 0 000001 0 -> 0x02
    assert slice_stream[3] == 0x02, "Lỗi: Slice Header không đánh dấu đúng TRAIL_R NAL Type!"

def test_full_stream_generation():
    """Kiểm tra hàm viết nguyên một Video Stream hoàn chỉnh."""
    packager = NalPackager(width=640, height=480)
    frames = [
        {"cabac_bytes": b'\xFF', "poc": 0, "is_idr": True, "slice_type": "I"},
        {"cabac_bytes": b'\xEE', "poc": 1, "is_idr": False, "slice_type": "P"}
    ]
    
    stream = packager.write_stream(frames)
    
    # Đếm số lượng START_CODE_4
    count_4 = stream.count(START_CODE_4)
    
    # Vì hàm count() đếm luôn cả đuôi của START_CODE_4, 
    # ta phải TRỪ ĐI count_4 để ra số lượng START_CODE_3 độc lập thực sự!
    count_3_only = stream.count(START_CODE_3) - count_4
    
    # VPS(4) + SPS(4) + PPS(4) + IDR(4) + P(3) => 4 cái 4-byte, 1 cái 3-byte độc lập
    assert count_4 == 4, "Lỗi số lượng Start Code 4-byte"
    assert count_3_only == 1, "Lỗi số lượng Start Code 3-byte"