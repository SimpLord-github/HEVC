import pytest
import numpy as np

# Import từ các module của bạn
from intra_prediction import (
    predict_intra_luma,
    predict_intra_chroma,
    generate_residual,
    full_intra_pipeline,
    CHROMA_DM,
    CHROMA_PLANAR
)
from intra_estimation import PLANAR_MODE, DC_MODE

# ---------------------------------------------------------------------------
# 1. TEST BỘ LỌC LÀM MƯỢT GÓC DC (DC Boundary Post-Filter)
# ---------------------------------------------------------------------------
def test_dc_post_filter_activation():
    """
    Chuẩn HEVC quy định: Với DC Mode ở các khối nhỏ hơn 32x32, 
    hàng đầu tiên và cột đầu tiên phải được lọc (blend) với viền tham chiếu.
    Ta test bằng cách chạy 2 lần: Có lọc và Không lọc, sau đó so sánh.
    """
    size = 8
    # Giả lập viền tham chiếu màu cực sáng (200), trong khi trung bình DC sẽ thấp hơn
    ref_above = np.full(2 * size + 1, 200, dtype=np.int16)
    ref_left = np.full(2 * size + 1, 100, dtype=np.int16)
    
    # Chạy lần 1: TẮT post-filter (Ảnh sẽ phẳng lỳ hoàn toàn)
    pred_no_filter = predict_intra_luma(
        DC_MODE, ref_above, ref_left, n=size, 
        filter_refs=False, apply_dc_filter=False
    )
    
    # Chạy lần 2: BẬT post-filter (Mặc định của HEVC)
    pred_with_filter = predict_intra_luma(
        DC_MODE, ref_above, ref_left, n=size, 
        filter_refs=False, apply_dc_filter=True
    )
    
    # Pixel ở góc trên cùng bên trái [0, 0] bắt buộc phải khác nhau
    # CORRECT — first row [0,1] and first col [1,0] are the actual filter targets
    assert pred_no_filter[0, 1] != pred_with_filter[0, 1], \
        "DC Post-filter không làm thay đổi pixel hàng đầu!"
    assert pred_no_filter[1, 0] != pred_with_filter[1, 0], \
        "DC Post-filter không làm thay đổi pixel cột đầu!"
    assert pred_no_filter[4, 4] == pred_with_filter[4, 4], \
        "DC Post-filter lan nhầm vào giữa khối ảnh!"
        
    # Pixel ở giữa khối [4, 4] không được bị ảnh hưởng bởi Post-filter
    assert pred_no_filter[4, 4] == pred_with_filter[4, 4], \
        "Lỗi: DC Post-filter lan nhầm vào giữa khối ảnh!"

# ---------------------------------------------------------------------------
# 2. TEST CHROMA INTRA PREDICTION (Đặc sản của video có màu)
# ---------------------------------------------------------------------------
def test_chroma_dm_mode():
    """
    Test chế độ Derived Mode (DM - Mode 4) của Chroma.
    Khối Chroma (Cb/Cr) sẽ 'copy' y hệt mode của khối Luma (Y) tương ứng.
    """
    size_c = 4  # Chroma block 4x4 (Tương ứng với Luma 8x8 trong video 4:2:0)
    ref_above_c = np.full(2 * size_c + 1, 128, dtype=np.int16)
    ref_left_c = np.full(2 * size_c + 1, 128, dtype=np.int16)
    
    # Nếu gọi DM mà 'quên' truyền luma_mode, hàm phải báo lỗi ngay
    with pytest.raises(ValueError, match="luma_mode must be provided"):
        predict_intra_chroma(CHROMA_DM, ref_above_c, ref_left_c, n_c=size_c)
        
    # Cấp luma_mode = 26 (Vertical). Khối Chroma cũng phải sinh ra ảnh dạng Vertical
    pred_chroma = predict_intra_chroma(
        CHROMA_DM, ref_above_c, ref_left_c, n_c=size_c, luma_mode=26
    )
    
    assert pred_chroma.shape == (size_c, size_c)

# ---------------------------------------------------------------------------
# 3. TEST TÍNH TOÁN RESIDUAL (Ngăn chặn Underflow)
# ---------------------------------------------------------------------------
def test_generate_residual_safety():
    """
    Trừ 2 ảnh uint8 (0-255) rất dễ sinh ra lỗi Underflow (ví dụ 10 - 20 = 246).
    Hàm generate_residual phải cast sang int16 trước khi trừ.
    """
    original = np.array([[10, 200]], dtype=np.uint8)
    pred = np.array([[20, 100]], dtype=np.uint8)
    
    residual = generate_residual(original, pred)
    
    assert residual.dtype == np.int16
    assert residual[0, 0] == -10  # Nếu lỗi underflow, chỗ này sẽ ra 246
    assert residual[0, 1] == 100

# ---------------------------------------------------------------------------
# 4. TEST TÍCH HỢP TOÀN BỘ PIPELINE (Full Pipeline Integration)
# ---------------------------------------------------------------------------
def test_full_intra_pipeline_integration():
    """
    Test ráp nối hoàn chỉnh: Ảnh gốc -> Luma Prediction -> Residual -> DCT -> Quantizer.
    Đầu ra phải là một cấu trúc dữ liệu IntraResult chứa đầy đủ các tầng.
    """
    size = 8
    np.random.seed(42)
    # Ảnh gốc ngẫu nhiên
    block = np.random.randint(50, 200, (size, size), dtype=np.uint8)
    ref_above = np.full(2 * size + 1, 128, dtype=np.int16)
    ref_left = np.full(2 * size + 1, 128, dtype=np.int16)
    
    QP = 28
    
    # Khởi chạy cỗ máy
    result = full_intra_pipeline(
        block, mode=PLANAR_MODE, ref_above=ref_above, ref_left=ref_left, qp=QP
    )
    
    # Kiểm tra độ trọn vẹn của cấu trúc dữ liệu đầu ra
    assert result.mode == PLANAR_MODE
    assert result.pred.shape == (size, size)
    assert result.residual.shape == (size, size)
    assert result.dct_coeffs is not None and result.dct_coeffs.shape == (size, size)
    assert result.quant_levels is not None and result.quant_levels.dtype == np.int32