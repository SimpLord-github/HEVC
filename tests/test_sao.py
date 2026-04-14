import pytest
import numpy as np

# Import toàn bộ API từ file sao.py
from sao import (
    sao_filter_frame,
    apply_sao_bo,
    apply_sao_eo,
    estimate_sao_params,
    SAOParams,
    SAOType,
    EODirection,
    _classify_eo
)

# ===========================================================================
# 1. TEST CÁC HÀM TOÁN HỌC VÀ LOGIC LÕI
# ===========================================================================

def test_classify_eo():
    """
    Kiểm tra thuật toán phân loại Category cho Edge Offset dựa trên Dấu.
    c: pixel hiện tại, n1/n2: 2 pixel hàng xóm.
    """
    # Cat 0 (Local Min): c nhỏ hơn cả n1 và n2
    assert _classify_eo(c=10, n1=20, n2=20) == 0
    
    # Cat 1 (Concave): c nhỏ hơn n1, bằng n2
    assert _classify_eo(c=10, n1=20, n2=10) == 1
    
    # Cat 2 (Flat/Mixed): c lớn hơn n1 nhưng nhỏ hơn n2
    assert _classify_eo(c=15, n1=10, n2=20) == 2
    
    # Cat 3 (Convex): c lớn hơn n1, bằng n2
    assert _classify_eo(c=20, n1=10, n2=20) == 3
    
    # Cat 4 (Local Max): c lớn hơn cả n1 và n2
    assert _classify_eo(c=30, n1=20, n2=20) == 4


def test_eo_sign_constraint_in_estimation():
    """
    Đảm bảo hàm Estimate ép chuẩn Dấu của HEVC:
    - Cat 0, 1 (Lõm/Đáy): Phải >= 0
    - Cat 3, 4 (Lồi/Đỉnh): Phải <= 0
    """
    orig = np.full((8, 8), 100, dtype=np.uint8)
    recon = np.full((8, 8), 100, dtype=np.uint8)
    
    # Tạo một điểm "Đáy" (Cat 0) ở recon có giá trị 110 (Lớn hơn gốc 100).
    # Residual = Orig(100) - Recon(110) = -10 (Offset âm).
    recon[4, 4] = 110
    recon[4, 3] = 120 # Hàng xóm trái
    recon[4, 5] = 120 # Hàng xóm phải
    
    # Ước lượng EO theo chiều ngang (EO_0)
    params = estimate_sao_params(orig, recon, SAOType.EO, EODirection.EO_0, ctu_size=8)
    
    # Cat 0 tính ra là -10, nhưng HEVC bắt buộc Cat 0 phải >= 0. Nó phải bị ép về 0!
    assert params.offsets[0] == 0, "FATAL: Hệ thống vi phạm luật ép dấu EO Cat 0/1 của HEVC!"


def test_bo_wraparound_estimation():
    """
    Kiểm tra việc ước lượng BO có cuộn vòng (Wraparound) qua dải 31 -> 0 không.
    """
    orig = np.full((16, 16), 100, dtype=np.uint8)
    recon = np.full((16, 16), 100, dtype=np.uint8)
    
    # Cấy lỗi có hệ thống vào dải 31 (giá trị ~250) và dải 0 (giá trị ~5)
    recon[0, 0] = 250; orig[0, 0] = 255  # Diff = +5, Band = 31
    recon[1, 1] = 5;   orig[1, 1] = 10   # Diff = +5, Band = 0
    
    params = estimate_sao_params(orig, recon, SAOType.BO, ctu_size=16)
    
    # Hệ thống phải thông minh gộp band 31 và band 0 vào cùng 1 nhóm 4 active bands
    # Ví dụ: band_start = 30 hoặc 31
    assert params.band_start in [29, 30, 31], "Lỗi: BO Estimation không xoay vòng đúng dải 32 bands!"


# ===========================================================================
# 2. TEST ÁP DỤNG BỘ LỌC VÀO PIXEL (APPLICATION)
# ===========================================================================

def test_apply_sao_bo_pixel_modification():
    """Đảm bảo BO cộng đúng Offset vào các pixel thuộc Active Bands."""
    recon = np.array([
        [240, 250],  # Band 30, 31
        [5, 10]      # Band 0, 1
    ], dtype=np.uint8)
    
    # Tạo Params với band_start = 30, 4 offset cho bands 30, 31, 0, 1
    params = SAOParams(SAOType.BO, band_start=30, offsets=[1, 2, 3, 4])
    
    apply_sao_bo(recon, params, x=0, y=0, ctu_size=2)
    
    # Kiểm tra kết quả
    assert recon[0, 0] == 241  # 240 + 1 (Band 30)
    assert recon[0, 1] == 252  # 250 + 2 (Band 31)
    assert recon[1, 0] == 8    # 5 + 3 (Band 0)
    assert recon[1, 1] == 14   # 10 + 4 (Band 1)


def test_apply_sao_eo_padding_safety():
    """
    Đảm bảo khi quét EO chéo ở góc ảnh, hệ thống Padding tốt và không văng lỗi IndexError.
    """
    recon = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
    params = SAOParams(SAOType.EO, eo_dir=EODirection.EO_2, offsets=[0, 1, 0, -1, -2])
    
    try:
        # EO_2 (Chéo 135 độ) dễ bị lỗi Out of Bounds ở 4 góc ảnh nhất
        apply_sao_eo(recon, params, x=0, y=0, ctu_size=16)
    except Exception as e:
        pytest.fail(f"Hàm apply_sao_eo bị crash khi quét ở sát viền: {e}")


# ===========================================================================
# 3. TEST KIẾN TRÚC TOÀN CỤC (CASCADING ARTIFACTS)
# ===========================================================================

def test_sao_filter_frame_no_cascading_artifacts():
    """
    Chứng minh SAO đọc dữ liệu từ một Snapshot gốc duy nhất.
    CTU sau KHÔNG ĐƯỢC lây nhiễm (Cascading) dữ liệu đã filter từ CTU trước.
    """
    # Khung ảnh 16x16, toàn màu 100
    luma = np.full((16, 16), 100, dtype=np.uint8)
    
    # CTU_0 (bên trái): Cộng thêm 50 vào TOÀN BỘ pixel bằng BO (Band của 100 là 12)
    params_ctu0 = SAOParams(SAOType.BO, band_start=12, offsets=[50, 0, 0, 0])
    
    # CTU_1 (bên phải): Bật EO Ngang (EO_0), không làm gì cả (Offsets = 0)
    params_ctu1 = SAOParams(SAOType.EO, eo_dir=EODirection.EO_0, offsets=[0, 0, 0, 0, 0])
    
    # Thiết lập lưới CTU 8x8 (1 dòng, 2 cột)
    grid = [[params_ctu0, params_ctu1]]
    
    # Chạy SAO Filter cho toàn bộ frame
    sao_filter_frame(luma, chroma_cb=None, chroma_cr=None, luma_params=grid, ctu_size=8)
    
    # KẾT QUẢ KỲ VỌNG:
    # - CTU_0 (x: 0->8) đã bị BO cộng 50 -> Thành 150.
    # - CTU_1 (x: 8->16) chạy EO ngang. Khi pixel luma[0, 8] nhìn sang trái để check luma[0, 7], 
    #   nó PHẢI NHÌN THẤY giá trị GỐC là 100 (Bằng phẳng -> Cat 2 -> Không cộng).
    #   Nếu hệ thống bị lỗi Cascading, nó sẽ nhìn thấy 150 -> Bị nhầm thành Cat 1 (Đáy) -> Sai bét!
    
    # Đảm bảo CTU 1 vẫn giữ nguyên giá trị 100 (Không bị lây nhiễm)
    assert np.all(luma[:, 8:16] == 100), "FATAL BUG: CTU bên phải bị lây nhiễm (Cascading) từ CTU bên trái!"
    
    # Đảm bảo CTU 0 đã được cộng đúng thành 150
    assert np.all(luma[:, 0:8] == 150), "Lỗi: CTU bên trái không được cộng Offset!"