import pytest
import numpy as np

from motion_compensation import (
    compensate_luma,
    compensate_chroma,
    compensate_bi,
    full_inter_pipeline,
    MotionVector
)

# ---------------------------------------------------------------------------
# 1. TEST LUMA COMPENSATION (Integer & Sub-pel)
# ---------------------------------------------------------------------------
def test_luma_integer_compensation():
    """
    Test MV chẵn (Quarter-pel chia hết cho 4).
    Hệ thống phải cắt chính xác ảnh (copy) mà không làm biến dạng qua filter.
    """
    n = 8
    ox, oy = 10, 10
    
    # Tạo frame tham chiếu (size 32x32)
    ref_frame = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    
    # MV dịch chuyển sang phải 2 pixel, xuống 1 pixel (ở hệ qpel: dx=8, dy=4)
    mv = MotionVector(mvx=8, mvy=4, sad_cost=0, satd_cost=0, ref_idx=0, algorithm="hex")
    
    # Kết quả kỳ vọng: cắt thẳng từ tọa độ (10+2, 10+1) = (12, 11)
    expected_patch = ref_frame[11:11+n, 12:12+n]
    
    # Khối bù trừ
    pred_patch = compensate_luma(ref_frame, origin=(ox, oy), mv=mv, n=n)
    
    # Bộ lọc 8-tap tại frac=0 phải trả về đúng y hệt bản gốc (Copy filter)
    np.testing.assert_array_equal(pred_patch, expected_patch)

def test_luma_subpel_pipeline_runs():
    """Test đảm bảo MV thập phân (sub-pel) chạy mượt mà qua bộ lọc 8-tap."""
    n = 16
    ref_frame = np.random.randint(0, 255, (40, 40), dtype=np.uint8)
    
    # MV: dịch nửa pixel (half-pel, qpel=2)
    mv = MotionVector(mvx=2, mvy=-2, sad_cost=0, satd_cost=0, ref_idx=0, algorithm="hex")
    
    pred_patch = compensate_luma(ref_frame, origin=(15, 15), mv=mv, n=n)
    
    assert pred_patch.shape == (n, n)
    assert pred_patch.dtype == np.uint8

# ---------------------------------------------------------------------------
# 2. TEST CHROMA COMPENSATION (Scale MV và 4-tap Filter)
# ---------------------------------------------------------------------------
def test_chroma_compensation_scaling():
    """
    Video 4:2:0 -> Khối Chroma kích thước bằng một nửa (N/2).
    Luma MV (qpel) phải được tự động chia đôi thành Chroma MV (1/8-pel).
    """
    n_c = 4  # Chroma block 4x4 (ứng với Luma 8x8)
    ox_c, oy_c = 5, 5
    
    ref_cb = np.full((20, 20), 100, dtype=np.uint8)
    ref_cr = np.full((20, 20), 150, dtype=np.uint8)
    
    # Đặt một "điểm neo" màu trắng vào vị trí đích của Cb
    # Luma MV=(8, 8) qpel -> Luma dịch 2 pixel
    # Do đó, Chroma phải dịch 1 pixel. Điểm gốc = 5, đích = 6.
    ref_cb[6:6+n_c, 6:6+n_c] = 200
    
    mv = MotionVector(mvx=8, mvy=8, sad_cost=0, satd_cost=0, ref_idx=0, algorithm="hex")
    
    pred_cb, pred_cr = compensate_chroma(ref_cb, ref_cr, origin_c=(ox_c, oy_c), mv=mv, n_c=n_c)
    
    # pred_cb phải bám đúng vào cái khung màu trắng (200)
    assert np.all(pred_cb == 200), "Lỗi: MV Chroma không được scale tỷ lệ đúng (chia 2)."
    assert np.all(pred_cr == 150)

# ---------------------------------------------------------------------------
# 3. TEST BI-DIRECTIONAL COMPENSATION (B-Frame Averaging)
# ---------------------------------------------------------------------------
def test_bi_directional_averaging():
    """
    Test B-slice. Bù trừ 2 chiều L0 và L1.
    Kết quả dự đoán (Pred) phải là trung bình cộng của 2 patch nội suy.
    """
    n = 8
    origin = (10, 10)
    
    # Tạo 2 frame tham chiếu (Quá khứ và Tương lai)
    ref_l0 = np.full((32, 32), 100, dtype=np.uint8)
    ref_l1 = np.full((32, 32), 200, dtype=np.uint8)
    
    # MV đứng im (0,0) cho dễ test
    mv_0 = MotionVector(0, 0, 0, 0, 0, "full")
    mv_1 = MotionVector(0, 0, 0, 0, 1, "full")
    
    # L0 trả về mảng 100, L1 trả về mảng 200. Trung bình = (100 + 200 + 1) // 2 = 150.
    pred_bi = compensate_bi(ref_l0, ref_l1, origin, mv_l0=mv_0, mv_l1=mv_1, n=n)
    
    assert np.all(pred_bi == 150), f"Lỗi: Trung bình cộng 2 chiều sai, nhận {pred_bi[0,0]}"

# ---------------------------------------------------------------------------
# 4. TEST XỬ LÝ NGOẠI LỆ (Boundary Handling)
# ---------------------------------------------------------------------------
def test_out_of_bounds_compensation():
    """Đảm bảo hệ thống bắt lỗi nếu MV đùn khối dự đoán ra khỏi khung hình tham chiếu."""
    ref_frame = np.zeros((32, 32), dtype=np.uint8)
    n = 16
    ox, oy = 20, 20
    # MV văng ra ngoài khung hình (dịch thêm 4 pixel = 16 qpel)
    mv = MotionVector(16, 16, 0, 0, 0, "full")
    
    with pytest.raises(ValueError, match="out of reference frame bounds"):
        compensate_luma(ref_frame, origin=(ox, oy), mv=mv, n=n)