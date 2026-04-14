import pytest
import numpy as np

from intra_estimation import (
    estimate_intra_mode,
    compute_sad,
    compute_satd,
    DC_MODE,
    PLANAR_MODE
)

# ---------------------------------------------------------------------------
# 1. TEST CÁC HÀM TÍNH CHI PHÍ (Cost Functions)
# ---------------------------------------------------------------------------
def test_sad_computation():
    """Kiểm tra hàm Sum of Absolute Differences (SAD)."""
    block1 = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    block2 = np.array([[12, 18], [35, 40]], dtype=np.uint8)
    
    # SAD = |10-12| + |20-18| + |30-35| + |40-40| = 2 + 2 + 5 + 0 = 9
    assert compute_sad(block1, block2) == 9

def test_satd_identical_blocks():
    """Hai block giống hệt nhau thì SATD (qua biến đổi Hadamard) phải bằng 0."""
    block = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
    assert compute_satd(block, block) == 0

# ---------------------------------------------------------------------------
# 2. TEST KHẢ NĂNG DÒ TÌM HƯỚNG DỰ ĐOÁN (Directional Mode Detection)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("size", [4, 8, 16])
def test_detect_vertical_mode(size):
    """
    Tạo ra một block có các sọc dọc (Vertical stripes).
    Thuật toán RMD + RDO phải tự động chốt Mode 26 (Vertical).
    """
    # 1. Tạo block gốc: Các cột có giá trị thay đổi, nhưng trong cùng 1 cột thì pixel y hệt nhau
    block = np.zeros((size, size), dtype=np.uint8)
    for col in range(size):
        block[:, col] = 50 + col * 10
        
    # 2. Dữ liệu tham chiếu (Reference): Phải khớp với mẫu sọc dọc này
    # Độ dài ref = 2*size + 1
    ref_above = np.zeros(2 * size + 1, dtype=np.int16)
    # Gán hàng top y hệt như hàng đầu tiên của block
    ref_above[1:size+1] = block[0, :]
    # Phần dư thừa (top-right) cứ cho random hoặc lặp lại
    ref_above[size+1:] = block[0, -1] 
    
    ref_left = np.full(2 * size + 1, 128, dtype=np.int16) # Viền trái không quan trọng với sọc dọc
    
    # 3. Chạy Estimation
    result = estimate_intra_mode(block, ref_above, ref_left)
    
    assert result.mode == 26, f"Lỗi: Ảnh sọc dọc rành rành nhưng lại chọn mode {result.mode}"
    assert result.sad_cost == 0, "Lỗi: Dự đoán vertical hoàn hảo thì SAD phải bằng 0"

@pytest.mark.parametrize("size", [4, 8, 16])
def test_detect_horizontal_mode(size):
    """
    Tạo ra một block có các sọc ngang (Horizontal stripes).
    Thuật toán phải chốt Mode 18 (Horizontal).
    """
    block = np.zeros((size, size), dtype=np.uint8)
    for row in range(size):
        block[row, :] = 50 + row * 10
        
    ref_left = np.zeros(2 * size + 1, dtype=np.int16)
    ref_left[1:size+1] = block[:, 0]
    ref_left[size+1:] = block[-1, 0]
    
    ref_above = np.full(2 * size + 1, 128, dtype=np.int16)
    
    result = estimate_intra_mode(block, ref_above, ref_left)
    
    assert result.mode == 10, f"Lỗi: Ảnh sọc ngang phải chọn mode 10, nhưng lại chọn {result.mode}"
    assert result.sad_cost == 0

def test_detect_dc_mode():
    """
    Tạo ra một block phẳng lì (màu đồng nhất).
    Mode DC (1) hoặc Planar (0) phải được ưu tiên.
    """
    size = 8
    # Block toàn màu 100
    block = np.full((size, size), 100, dtype=np.uint8)
    
    # Viền xung quanh cũng màu 100
    ref_above = np.full(2 * size + 1, 100, dtype=np.int16)
    ref_left = np.full(2 * size + 1, 100, dtype=np.int16)
    
    result = estimate_intra_mode(block, ref_above, ref_left)
    
    # Với ảnh phẳng hoàn toàn, cả DC và Planar đều cho SAD = 0.
    # Thông thường thuật toán sẽ ưu tiên mode có index nhỏ nhất (hoặc do code sắp xếp)
    assert result.mode in [DC_MODE, PLANAR_MODE], f"Lỗi: Ảnh phẳng nên chọn DC hoặc Planar, đang chọn {result.mode}"
    assert result.sad_cost == 0

# ---------------------------------------------------------------------------
# 3. TEST XỬ LÝ NGOẠI LỆ (Input Validation)
# ---------------------------------------------------------------------------
def test_invalid_reference_arrays():
    """Đảm bảo hệ thống từ chối các mảng reference có chiều dài sai chuẩn."""
    block = np.zeros((8, 8), dtype=np.uint8)
    
    # Đúng chuẩn n=8 thì ref phải dài 2*8 + 1 = 17
    invalid_ref_above = np.zeros(16, dtype=np.int16) # Thiếu 1 pixel
    valid_ref_left = np.zeros(17, dtype=np.int16)
    
    with pytest.raises(ValueError, match="ref_above must be 1-D with length 17"):
        estimate_intra_mode(block, invalid_ref_above, valid_ref_left)