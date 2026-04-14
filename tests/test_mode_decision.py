import pytest
import numpy as np
import math

from mode_decision import (
    decide_mode,
    compute_lambda,
    rd_cost,
    MODE_INTRA,
    MODE_INTER,
    MODE_SKIP
)

# ---------------------------------------------------------------------------
# 1. TEST CÁC HÀM TOÁN HỌC (Math Helpers)
# ---------------------------------------------------------------------------
def test_compute_lambda():
    """Test công thức tính hệ số Lagrange chuẩn x265."""
    # Tại QP=27: lambda = 0.57 * 2^((27-12)/3) = 0.57 * 2^5 = 0.57 * 32 = 18.24
    lam = compute_lambda(27)
    assert math.isclose(lam, 18.24, rel_tol=1e-5)
    
    # QP tăng, lambda phải tăng (ưu tiên tiết kiệm bit hơn chất lượng)
    assert compute_lambda(30) > compute_lambda(27)

def test_rd_cost():
    """Test hàm J = D + lambda * R"""
    distortion = 100
    rate = 10
    lam = 5.0
    # J = 100 + 5.0 * 10 = 150.0
    assert math.isclose(rd_cost(distortion, rate, lam), 150.0)

# ---------------------------------------------------------------------------
# 2. TEST LOGIC RDO (Trọng tài Mode)
# ---------------------------------------------------------------------------
def test_rdo_chooses_skip():
    """
    Tạo bẫy SKIP: Khung hiện tại giống hệt khung tham chiếu tại đúng vị trí đó.
    RDO bắt buộc phải chọn SKIP để tối ưu dung lượng (chỉ tốn 1 bit).
    """
    n = 8
    origin = (10, 10)
    
    # Khung tham chiếu và Block y hệt nhau (Màu 100)
    ref_frame = np.full((32, 32), 100, dtype=np.uint8)
    block = np.full((n, n), 100, dtype=np.uint8)
    
    # Intra reference là nhiễu (Intra sẽ cho SSE rất cao)
    ref_above = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    ref_left = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    
    decision = decide_mode(block, ref_above, ref_left, ref_frame, origin, qp=28, slice_type="P")
    
    assert decision.mode == MODE_SKIP, "Lỗi: Đáng lẽ phải chọn SKIP nhưng lại chọn mode khác."
    assert decision.distortion == 0
    assert decision.rate_proxy == 1  # 1 bit cho skip flag

def test_rdo_chooses_intra():
    """
    Tạo bẫy INTRA: Khung tham chiếu hoàn toàn là nhiễu rác, không thể bù trừ chuyển động.
    Nhưng block hiện tại lại có viền trên và trái giống hệt nó (phù hợp Intra DC/Planar).
    RDO bắt buộc phải chọn INTRA.
    """
    n = 8
    origin = (10, 10)
    
    block = np.full((n, n), 200, dtype=np.uint8)
    
    # Khung tham chiếu (Inter) là nhiễu rác (SSE sẽ rất lớn)
    ref_frame = np.random.randint(0, 50, (32, 32), dtype=np.uint8)
    
    # Viền tham chiếu (Intra) lý tưởng
    ref_above = np.full(2*n + 1, 200, dtype=np.int16)
    ref_left = np.full(2*n + 1, 200, dtype=np.int16)
    
    decision = decide_mode(block, ref_above, ref_left, ref_frame, origin, qp=28, slice_type="P")
    
    assert decision.mode == MODE_INTRA, "Lỗi: Đáng lẽ phải chọn INTRA do Inter quá nhiễu."
    assert decision.is_intra is True

def test_rdo_chooses_inter():
    """
    Tạo bẫy INTER: Viền Intra là nhiễu rác. 
    Trong khi đó, khung tham chiếu có một mảng giống hệt block nhưng bị dịch chuyển.
    """
    n = 8
    origin = (10, 10)
    dx, dy = 3, -2
    
    block = np.full((n, n), 150, dtype=np.uint8)
    
    # Khung tham chiếu bình thường là màu đen (0)
    ref_frame = np.zeros((32, 32), dtype=np.uint8)
    # Cấy block 150 vào vị trí bị dịch chuyển
    ref_frame[origin[1]+dy : origin[1]+dy+n, origin[0]+dx : origin[0]+dx+n] = 150
    
    # Viền Intra là nhiễu
    ref_above = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    ref_left = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    
    decision = decide_mode(block, ref_above, ref_left, ref_frame, origin, qp=28, slice_type="P")
    
    # Inter phải tìm ra (dx=3, dy=-2) và chiến thắng
    assert decision.mode == MODE_INTER, "Lỗi: Đáng lẽ phải chọn INTER."
    assert decision.mv.mvx == dx * 4
    assert decision.mv.mvy == dy * 4

# ---------------------------------------------------------------------------
# 3. TEST I-SLICE (Chặn Inter)
# ---------------------------------------------------------------------------
def test_slice_i_forces_intra():
    """Ở slice I, hệ thống cấm dùng Inter/Skip dù khung tham chiếu có tốt thế nào."""
    n = 8
    origin = (10, 10)
    block = np.full((n, n), 100, dtype=np.uint8)
    ref_frame = np.full((32, 32), 100, dtype=np.uint8) # Bẫy SKIP hoàn hảo
    
    ref_above = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    ref_left = np.random.randint(0, 255, 2*n + 1, dtype=np.int16)
    
    # Gọi với slice_type = "I"
    decision = decide_mode(block, ref_above, ref_left, ref_frame, origin, qp=28, slice_type="I")
    
    assert decision.mode == MODE_INTRA, "Lỗi: I-Slice không được phép chọn SKIP/INTER!"