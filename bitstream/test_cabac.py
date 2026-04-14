import pytest
import numpy as np
from cabac import (
    CABACEncoder,
    _last_coord_prefix,
    _prefix_to_min_coord
)

# ---------------------------------------------------------------------------
# 1. TEST BYPASS RENORMALIZATION (CHỐNG LỖI CORRUPT BITSTREAM)
# ---------------------------------------------------------------------------
def test_bypass_renormalization_leak():
    """
    Ép CABAC vào trạng thái _low nằm trong [1<<16, 1<<17).
    Nếu thiếu khối `else`, _low sẽ phình to và giá trị sinh ra bị sai hoàn toàn.
    """
    enc = CABACEncoder(qp=28)
    
    # Mã hóa liên tục một chuỗi các bit 1 và 0 đan xen bằng bypass
    # Điều này chắc chắn sẽ đẩy _low vào trạng thái kẹt (bits_outstanding tăng)
    for i in range(50):
        enc._encode_bypass(i % 2)
        
    enc.flush()
    stream = enc.get_bits()
    
    # Nếu code cũ (không có else), đoạn mã này sẽ sinh byte rác hoặc crash.
    # Code mới sẽ đẩy các bit ra một cách an toàn và đúng chiều dài.
    assert len(stream) > 0, "CABAC Encoder failed to produce bytes!"
    # 50 bits bypass + 1 bit stop + RBSP = khoảng 7 byte
    assert len(stream) == 7, f"Lỗi rò rỉ thanh ghi: Số byte tạo ra bị sai ({len(stream)})"

# ---------------------------------------------------------------------------
# 2. TEST HEVC TABLE 9-42 (LAST_SIG_COEFF BINARISATION)
# ---------------------------------------------------------------------------
def test_hevc_last_sig_coeff_binarisation():
    """
    Kiểm chứng chuẩn xác bảng ánh xạ Binarization của HEVC.
    """
    # Các toạ độ từ 0 đến 15
    coords = list(range(16))
    
    # Kết quả Prefix KỲ VỌNG theo đúng chuẩn ISO/IEC 23008-2
    expected_prefixes = [
        0, 1, 2, 3,          # coord 0-3
        4, 4,                # coord 4-5
        5, 5,                # coord 6-7
        6, 6, 6, 6,          # coord 8-11
        7, 7, 7, 7           # coord 12-15
    ]
    
    # Tính toán qua hàm đã fix
    calculated_prefixes = [_last_coord_prefix(c) for c in coords]
    
    assert calculated_prefixes == expected_prefixes, "Lỗi FATAL: Thuật toán tính Prefix vi phạm chuẩn HEVC!"
    
    # Kiểm tra hàm Min Coord
    assert _prefix_to_min_coord(4) == 4
    assert _prefix_to_min_coord(5) == 6
    assert _prefix_to_min_coord(6) == 8
    assert _prefix_to_min_coord(7) == 12

# ---------------------------------------------------------------------------
# 3. TEST TOÀN BỘ PIPELINE ENCODING VÀ FLUSH
# ---------------------------------------------------------------------------
def test_full_cabac_encoding_pipeline():
    """Đảm bảo cỗ máy CABAC không bị crash khi nén dữ liệu thực tế."""
    enc = CABACEncoder(qp=28)
    
    # Giả lập nén 1 CU
    enc.encode_split_flag(True, depth=0)
    enc.encode_pred_mode_flag(is_intra=True)
    enc.encode_intra_luma_mode(mode=15, mpm_list=[0, 1, 26])
    enc.encode_cbf(True, is_luma=True, tu_depth=0)
    
    # Giả lập 1 mảng hệ số lượng tử (TU 4x4)
    sig_map = np.zeros((4, 4), dtype=bool)
    levels  = np.zeros((4, 4), dtype=np.int32)
    sig_map[0,0] = True; levels[0,0] = 5
    sig_map[0,1] = True; levels[0,1] = -2
    
    # Báo cáo vị trí cuối cùng có dữ liệu là (0, 1)
    enc.encode_last_sig_pos(0, 1, log2_tu_size=2)
    enc.encode_sig_coeff_map(sig_map, log2_tu_size=2)
    enc.encode_coeff_levels(levels, sig_map, log2_tu_size=2, is_luma=True)
    
    enc.flush()
    bits = enc.get_bit_count()
    
    assert bits > 0, "Lỗi: Pipeline nén CABAC không sinh ra được bit nào!"