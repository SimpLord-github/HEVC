import pytest
import numpy as np

# Giả định deblocking.py ở thư mục gốc hoặc bạn đã config đúng sys.path
from deblocking import (
    deblock_frame,
    compute_beta,
    compute_tc,
    compute_boundary_strength,
)

# ---------------------------------------------------------------------------
# 1. TEST BẢNG TRA CỨU TIÊU CHUẨN HEVC
# ---------------------------------------------------------------------------
def test_table_lookups():
    """Kiểm tra các giá trị biên của bảng β và tC."""
    assert compute_beta(0) == 0
    assert compute_beta(51) == 64
    assert compute_tc(0, bs=1) == 0
    assert compute_tc(53, bs=1) == 56

def test_compute_boundary_strength():
    """BS phải được gán đúng theo chuẩn HEVC."""
    # Nếu có khối Intra -> BS = 2
    assert compute_boundary_strength(28, 28, intra_p=True) == 2
    # Cả 2 là Inter, nhưng có dư (residual) -> BS = 1
    assert compute_boundary_strength(28, 28, cbf_p=True) == 1
    # Inter hoàn hảo (trùng MV, không có residual) -> BS = 0
    assert compute_boundary_strength(28, 28, cbf_p=False, cbf_q=False) == 0

# ---------------------------------------------------------------------------
# 2. TEST MỨC ĐỘ LÀM MƯỢT CỦA WEAK FILTER (LUMA)
# ---------------------------------------------------------------------------
def test_weak_luma_filtering_smooths_edge():
    """
    Tạo một bức ảnh 16x16. Nửa trái màu 100, nửa phải màu 120.
    Nếu không lọc, sự chênh lệch tại biên (x=8) là 20 đơn vị (răng cưa rất gắt).
    Sau khi lọc, sự chênh lệch này phải giảm xuống để mắt người thấy mượt hơn.
    """
    # Khởi tạo ảnh 16x16
    luma = np.zeros((16, 16), dtype=np.uint8)
    luma[:, :8] = 100  # Nửa trái
    luma[:, 8:] = 120  # Nửa phải

    # Tính độ gắt (Gradient) trước khi lọc tại dòng đầu tiên
    diff_before = abs(int(luma[0, 8]) - int(luma[0, 7]))
    assert diff_before == 20

    # Chạy qua cỗ máy Deblocking (QP=35 để bộ lọc kích hoạt mạnh một chút)
    deblock_frame(luma, chroma_cb=None, chroma_cr=None, qp=35)

    # Tính độ gắt sau khi lọc
    diff_after = abs(int(luma[0, 8]) - int(luma[0, 7]))
    
    # Răng cưa phải giảm (diff_after < diff_before)
    assert diff_after < diff_before, "Lỗi: Deblocking filter không làm mượt được cạnh!"

# ---------------------------------------------------------------------------
# 3. TEST AN TOÀN VIỀN ẢNH (OUT-OF-BOUNDS)
# ---------------------------------------------------------------------------
def test_deblock_frame_no_crash_on_edges():
    """
    Đảm bảo khi quét tới sát viền (Ví dụ ảnh kích thước lẻ hoặc nhỏ), 
    vòng lặp không bị văng lỗi IndexError do thiếu pixel để làm mồi cho p[3], q[3].
    """
    # Kích thước 20x20 sẽ khiến vòng lặp chạm vào lề (không tròn block 8x8)
    luma = np.random.randint(0, 255, (20, 20), dtype=np.uint8)
    cb   = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    
    # Nếu code bắt OOB kém, dòng này sẽ Crash ngay lập tức!
    try:
        deblock_frame(luma, chroma_cb=cb, chroma_cr=cb, qp=28)
    except Exception as e:
        pytest.fail(f"Deblocking bị Crash khi chạm lề ảnh: {e}")