import pytest
import numpy as np

from motion_estimation import (
    estimate_motion,
    ALGO_HEX,
    ALGO_FULL,
    _get_filter_coeffs,
    interpolate_half_pel,
    interpolate_qpel
)

# ---------------------------------------------------------------------------
# 1. TEST TÌM KIẾM THEO SỐ NGUYÊN (Integer-Pel Search)
# ---------------------------------------------------------------------------
def test_zero_motion():
    """Nếu block giống hệt khung tham chiếu tại đúng vị trí, MV phải là (0,0)."""
    n = 16
    ox, oy = 20, 20
    np.random.seed(42)
    # Tạo một khung tham chiếu ngẫu nhiên
    ref_frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # Cắt block hiện tại từ đúng vị trí (20, 20)
    current_block = ref_frame[oy:oy+n, ox:ox+n].copy()
    
    # Dùng thuật toán HEX mặc định, tắt sub-pel refine để test raw integer
    mv = estimate_motion(current_block, ref_frame, origin=(ox, oy), use_satd_refine=False)
    
    assert mv.mvx == 0 and mv.mvy == 0, f"Lỗi: Ảnh đứng im mà MV lại là {mv.mvx}, {mv.mvy}"
    assert mv.sad_cost == 0

def test_integer_motion_tracking():
    """
    Dịch chuyển block đi một khoảng (dx=3, dy=-2).
    Hệ thống bắt buộc phải tìm ra MV tương ứng, nhân 4 lên thành (12, -8) ở đơn vị Quarter-pel.
    """
    n = 16
    ox, oy = 30, 30
    dx, dy = 3, -2
    
    ref_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    # Tạo một "vật thể" (hình vuông màu trắng) trong khung tham chiếu ở vị trí đã dịch chuyển
    ref_frame[oy+dy : oy+dy+n, ox+dx : ox+dx+n] = 200 
    
    # Ở frame hiện tại, vật thể đang nằm ở (ox, oy)
    current_block = np.full((n, n), 200, dtype=np.uint8)
    
    # Dùng FULL search để tránh việc HEX search rơi vào điểm mù cục bộ
    mv = estimate_motion(
        current_block, ref_frame, origin=(ox, oy), 
        algorithm=ALGO_FULL, use_satd_refine=False
    )
    
    # Nhắc lại: MV tính bằng Quarter-pel, nên 3 pixel = +12 qpel, -2 pixel = -8 qpel
    assert mv.mvx == dx * 4, f"Lỗi MVX: kỳ vọng {dx*4}, nhận {mv.mvx}"
    assert mv.mvy == dy * 4, f"Lỗi MVY: kỳ vọng {dy*4}, nhận {mv.mvy}"

# ---------------------------------------------------------------------------
# 2. TEST BỘ LỌC ĐỐI XỨNG (Filter Symmetry)
# ---------------------------------------------------------------------------
def test_filter_symmetry():
    """Đảm bảo phân số 3/4 (frac=6) là phiên bản lật ngược của 1/4 (frac=2)."""
    filter_1_4 = _get_filter_coeffs(2)
    filter_3_4 = _get_filter_coeffs(6)
    
    assert filter_3_4 == filter_1_4[::-1], "Lỗi: Bộ lọc HEVC luma chưa đối xứng!"

# ---------------------------------------------------------------------------
# 3. TEST TÍCH HỢP SUB-PEL REFINEMENT (Pipeline chạy ổn định)
# ---------------------------------------------------------------------------
def test_subpel_refinement_pipeline():
    """
    Test đảm bảo luồng tinh chỉnh Half-pel và Quarter-pel chạy mượt mà,
    gọi đúng các hàm nội suy và không bị văng lỗi (crash) Out-of-Bounds.
    """
    n = 8
    ox, oy = 10, 10
    
    # Khung tham chiếu lớn hơn một chút để filter 8-tap có dư dải lề nội suy
    ref_frame = np.random.randint(50, 200, (40, 40), dtype=np.uint8)
    current_block = np.random.randint(50, 200, (n, n), dtype=np.uint8)
    
    # BẬT use_satd_refine = True (mặc định)
    mv = estimate_motion(
        current_block, ref_frame, origin=(ox, oy), 
        algorithm=ALGO_HEX, use_satd_refine=True
    )
    
    # Chỉ cần hàm trả về kết quả hợp lệ (kiểu dữ liệu đúng) là đạt yêu cầu pipeline
    assert isinstance(mv.mvx, int)
    assert isinstance(mv.mvy, int)
    assert isinstance(mv.satd_cost, int)
    # MV không được vượt quá search range (32 pixel = 128 qpel)
    assert -128 <= mv.mvx <= 128
    assert -128 <= mv.mvy <= 128

# ---------------------------------------------------------------------------
# 4. TEST XỬ LÝ NGOẠI LỆ (Boundary Handling)
# ---------------------------------------------------------------------------
def test_out_of_bounds_origin():
    """Đảm bảo hệ thống chặn đứng việc tìm kiếm nếu Block nằm ngoài khung."""
    block = np.zeros((16, 16), dtype=np.uint8)
    ref_frame = np.zeros((32, 32), dtype=np.uint8)
    
    # Origin (20, 20) + n(16) = 36 > kích thước khung (32)
    with pytest.raises(ValueError, match="is outside reference frame"):
        estimate_motion(block, ref_frame, origin=(20, 20))