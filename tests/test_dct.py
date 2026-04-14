import pytest
import numpy as np
from dct import forward_dct, forward_dst

# ---------------------------------------------------------------------------
# 1. TEST KÍCH THƯỚC VÀ KIỂU DỮ LIỆU (Shape & Dtype)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_forward_dct_shapes_and_types(size):
    """Kiểm tra xem hàm có trả về đúng kích thước NxN và kiểu int32 không."""
    # Tạo block ngẫu nhiên với giá trị 16-bit (mô phỏng phần dư thực tế)
    residual = np.random.randint(-255, 255, (size, size), dtype=np.int16)
    coeffs = forward_dct(residual)
    
    assert coeffs.shape == (size, size), f"Lỗi: Kích thước đầu ra phải là {size}x{size}"
    assert coeffs.dtype == np.int32, "Lỗi: Kiểu dữ liệu đầu ra phải là int32 để chống tràn số"

def test_forward_dst_shape_and_type():
    """Kiểm tra riêng cho khối DST-7 4x4."""
    residual = np.random.randint(-255, 255, (4, 4), dtype=np.int16)
    coeffs = forward_dst(residual)
    
    assert coeffs.shape == (4, 4)
    assert coeffs.dtype == np.int32

# ---------------------------------------------------------------------------
# 2. TEST TÍNH CHẤT TOÁN HỌC (Mathematical Properties)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_zero_residual(size):
    """Một khối phần dư toàn số 0 phải cho ra hệ số toàn số 0."""
    residual = np.zeros((size, size), dtype=np.int16)
    coeffs = forward_dct(residual)
    
    assert np.all(coeffs == 0), "Lỗi: Khối toàn 0 phải sinh ra hệ số DCT toàn 0"

@pytest.mark.parametrize("size", [4, 8, 16, 32])
def test_dc_energy_compaction(size):
    """
    Test tính chất dồn năng lượng (Energy Compaction) của DCT.
    Một khối phẳng (Flat block - tất cả pixel bằng nhau) sẽ dồn toàn bộ 
    năng lượng vào hệ số DC (góc trên cùng bên trái [0, 0]), 
    các hệ số AC còn lại phải bằng 0 (hoặc xấp xỉ 0 do sai số làm tròn số nguyên).
    """
    residual = np.full((size, size), 100, dtype=np.int16)
    coeffs = forward_dct(residual)
    
    # Hệ số DC phải là số dương và rất lớn
    assert coeffs[0, 0] > 0, "Lỗi: Hệ số DC phải mang giá trị lớn với block phẳng"
    
    # Kiểm tra một hệ số AC ngẫu nhiên (ví dụ [size-1, size-1]) xem có bị triệt tiêu không
    assert coeffs[size-1, size-1] == 0, "Lỗi: Các hệ số AC tần số cao phải bằng 0 với block phẳng"

# ---------------------------------------------------------------------------
# 3. TEST XỬ LÝ NGOẠI LỆ (Exception Handling)
# ---------------------------------------------------------------------------
def test_invalid_shapes():
    """Đảm bảo hệ thống bắt được các lỗi truyền sai kích thước ma trận."""
    # 1. Truyền ma trận không vuông
    with pytest.raises(ValueError, match="Block must be square"):
        forward_dct(np.zeros((4, 8)))

    # 2. Truyền ma trận vuông nhưng không thuộc chuẩn HEVC (ví dụ 5x5)
    with pytest.raises(ValueError, match="Unsupported transform size"):
        forward_dct(np.zeros((5, 5)))

    # 3. Truyền mảng 1D hoặc 3D
    with pytest.raises(ValueError, match="Block must be 2-D"):
        forward_dct(np.zeros((4, 4, 4)))

def test_dst_invalid_size():
    """Khối DST-7 trong HEVC chỉ hỗ trợ 4x4."""
    with pytest.raises(ValueError, match="forward_dst requires a 4×4 block"):
        forward_dst(np.zeros((8, 8)))