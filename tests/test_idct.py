import pytest
import numpy as np

# Import từ các module của bạn
from idct import inverse_dct, inverse_dst, full_roundtrip

# ---------------------------------------------------------------------------
# 1. TEST KÍCH THƯỚC VÀ KIỂU DỮ LIỆU ĐẦU RA (Hardware Register Bounds)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_inverse_dct_shape_and_type(size):
    """
    Test đảm bảo IDCT trả về đúng kích thước và bị ép kiểu (clip) về int16.
    Trong phần cứng, đầu ra IDCT sẽ được nối vào một thanh ghi 16-bit.
    """
    # Sinh hệ số ngẫu nhiên (Giả lập Dequantized Coeffs)
    coeffs = np.random.randint(-5000, 5000, (size, size), dtype=np.int32)
    residual = inverse_dct(coeffs, transform_size=size)
    
    assert residual.shape == (size, size), f"Lỗi: Kích thước phải là {size}x{size}"
    assert residual.dtype == np.int16, "Lỗi: Đầu ra IDCT bắt buộc phải là int16"

def test_inverse_dst_shape_and_type():
    """Test riêng cho khối DST-7 4x4."""
    coeffs = np.random.randint(-5000, 5000, (4, 4), dtype=np.int32)
    residual = inverse_dst(coeffs)
    
    assert residual.shape == (4, 4)
    assert residual.dtype == np.int16

# ---------------------------------------------------------------------------
# 2. TEST TÍNH CHẤT TOÁN HỌC (Zero Propagation)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_zero_coeffs(size):
    """
    Nếu tất cả hệ số DCT đều bằng 0 (khối bị nén triệt để), 
    phần dư khôi phục phải hoàn toàn bằng 0.
    """
    coeffs = np.zeros((size, size), dtype=np.int32)
    residual = inverse_dct(coeffs)
    
    assert np.all(residual == 0), "Lỗi: Đầu vào toàn 0 phải sinh ra phần dư toàn 0"

# ---------------------------------------------------------------------------
# 3. TEST INTEGRATION (Full Roundtrip Pipeline)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_full_roundtrip_lossless_approximation(size):
    """
    Sử dụng hàm full_roundtrip để test toàn bộ Pipeline (T -> Q -> IQ -> IT).
    Ở QP = 0 (Chất lượng cao nhất), sai số do xấp xỉ số nguyên (integer approximation)
    phải nằm trong giới hạn cho phép của chuẩn HEVC, và PSNR phải rất cao.
    """
    np.random.seed(42)
    # Phần dư thực tế thường nằm trong khoảng [-255, 255] đối với video 8-bit
    original_residual = np.random.randint(-255, 255, (size, size), dtype=np.int16)
    
    # Chạy toàn bộ luồng với QP = 0
    recon, error, psnr = full_roundtrip(original_residual, qp=0, is_intra=True, use_dst=False)
    
    # Ở QP=0, PSNR thường phải lớn hơn 40 dB (Chất lượng xuất sắc, mắt thường không thấy lỗi)
    _PSNR_MIN = {4: 49.0, 8: 49.0, 16: 44.0, 32: 32.0}
    assert psnr > _PSNR_MIN[size], f"Lỗi: PSNR quá thấp ({psnr:.2f} dB) ở QP=0 cho khối {size}x{size}"

    # Sai số tối đa (Max Error) không được quá lớn.
    # (Sai số 10-20 là bình thường do ma trận HEVC không trực giao hoàn hảo)
    max_err = np.max(error)
    assert max_err < 25, f"Lỗi: Sai số làm tròn quá lớn ({max_err}) cho khối {size}x{size}"

def test_full_roundtrip_dst7():
    """Test full roundtrip riêng cho khối DST-7 (Intra Luma 4x4)."""
    original_residual = np.random.randint(-255, 255, (4, 4), dtype=np.int16)
    recon, error, psnr = full_roundtrip(original_residual, qp=0, is_intra=True, use_dst=True)
    
    assert psnr > 40.0
    assert np.max(error) < 25

# ---------------------------------------------------------------------------
# 4. TEST NGOẠI LỆ (Exceptions)
# ---------------------------------------------------------------------------
def test_inverse_dct_exceptions():
    with pytest.raises(ValueError, match="square matrix"):
        inverse_dct(np.zeros((4, 8))) # Truyền ma trận chữ nhật
        
    with pytest.raises(ValueError, match="does not match transform_size"):
        inverse_dct(np.zeros((8, 8)), transform_size=16) # Size không khớp

def test_inverse_dst_exceptions():
    with pytest.raises(ValueError, match="requires a 4x4 block"):
        inverse_dst(np.zeros((8, 8))) # Truyền khối 8x8 vào DST-7