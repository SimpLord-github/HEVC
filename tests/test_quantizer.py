import pytest
import numpy as np
from quantizer import quantize, dequantize, qp_to_step_size

# ---------------------------------------------------------------------------
# 1. TEST TÍNH NÉN (Dead-zone & Zeroing)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_dead_zone_zeroing(size):
    """
    Test Vùng chết (Dead-zone): Các hệ số nhỏ phải bị ép về 0.
    Với QP cao, ngay cả các hệ số tương đối lớn cũng phải biến mất.
    """
    # Khối chỉ có các chi tiết siêu nhỏ (nhiễu)
    coeffs = np.full((size, size), 8, dtype=np.int32)
    
    # Tại QP = 30, Qstep khoảng ~20. Các hệ số 15 chắc chắn rớt vào dead-zone
    levels = quantize(coeffs, qp=30, is_intra=False, transform_size=size)
    
    assert np.all(levels == 0), "Lỗi: Hệ số nhỏ chưa bị Dead-zone ép về 0"

# ---------------------------------------------------------------------------
# 2. TEST SỰ KHÁC BIỆT INTRA VS INTER
# ---------------------------------------------------------------------------
def test_intra_vs_inter_deadzone():
    """
    Do Intra có offset (2/3) lớn hơn Inter (1/2), 
    với cùng một hệ số ranh giới, Intra có thể giữ lại được mức lượng tử = 1
    trong khi Inter sẽ bóp nó về 0.
    """
    # Một hệ số "nhạy cảm" nằm ngay ranh giới làm tròn
    coeffs = np.array([[45, 45], [45, 45]], dtype=np.int32)
    QP = 20
    
    levels_intra = quantize(coeffs, qp=QP, is_intra=True, transform_size=4)
    levels_inter = quantize(coeffs, qp=QP, is_intra=False, transform_size=4)
    
    # Intra bảo tồn chi tiết tốt hơn Inter
    assert levels_intra[0, 0] >= levels_inter[0, 0], "Lỗi: Dead-zone của Intra và Inter hoạt động sai logic"

# ---------------------------------------------------------------------------
# 3. TEST QUÁ TRÌNH KHÔI PHỤC (Roundtrip Q -> IQ)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("qp", [0, 20, 40, 51])
def test_quantize_dequantize_roundtrip(qp):
    """
    Quá trình Q -> IQ là có mất mát (lossy), nhưng hệ số khôi phục
    phải bám sát (tỷ lệ thuận) với hệ số ban đầu.
    """
    np.random.seed(42) # Cố định seed để dễ debug
    # Tạo khối 8x8 với dải giá trị rộng (-1000 đến 1000)
    original_coeffs = np.random.randint(-1000, 1000, (8, 8), dtype=np.int32)
    
    levels = quantize(original_coeffs, qp=qp, is_intra=True, transform_size=8)
    recon_coeffs = dequantize(levels, qp=qp, transform_size=8)
    
    # Kích thước phải giữ nguyên
    assert recon_coeffs.shape == (8, 8)
    
    if qp == 0:
        # Tại QP = 0 (Chất lượng cao nhất), sai số (Error) phải rất nhỏ
        error = np.abs(original_coeffs - recon_coeffs)
        assert np.max(error) < 50, f"Lỗi: Tại QP=0 sai số quá lớn, max error = {np.max(error)}"
    elif qp == 51:
        # Tại QP = 51, hệ số ban đầu nhỏ hơn Qstep (~1140) sẽ bị triệt tiêu thành 0 hết
        assert np.count_nonzero(levels) < 64, "Lỗi: Ở QP cực đại, hình ảnh phải bị nén mạnh"

# ---------------------------------------------------------------------------
# 4. TEST XỬ LÝ NGOẠI LỆ
# ---------------------------------------------------------------------------
def test_invalid_qp():
    with pytest.raises(ValueError, match="QP must be in"):
        quantize(np.zeros((4,4)), qp=52)
        
    with pytest.raises(ValueError, match="QP must be in"):
        quantize(np.zeros((4,4)), qp=-1)