import pytest
import numpy as np

# Giả sử decoded_picture_buffer và reference_manager nằm cùng thư mục module (tùy chỉnh đường dẫn nếu cần)
from decoded_picture_buffer import DecodedPictureBuffer, FrameType, make_grey_frame
from reference_manager import ReferenceManager

@pytest.fixture
def populated_dpb():
    """Tạo một DPB chứa sẵn các frame 0 (I), 2 (B), 4 (P), 8 (P)."""
    dpb = DecodedPictureBuffer(max_buffering=5)
    dpb.store(make_grey_frame(0, 0, FrameType.I, 64, 64))
    dpb.store(make_grey_frame(2, 2, FrameType.B, 64, 64))
    dpb.store(make_grey_frame(4, 1, FrameType.P, 64, 64))
    dpb.store(make_grey_frame(8, 3, FrameType.P, 64, 64))
    return dpb

# ---------------------------------------------------------------------------
# 1. TEST XÂY DỰNG L0 (CHO P-SLICE VÀ B-SLICE)
# ---------------------------------------------------------------------------
def test_build_l0_p_slice(populated_dpb):
    """
    P-slice: L0 chỉ được chứa các khung quá khứ (preceding) và LTR. 
    Không bao giờ có khung tương lai (following).
    """
    rm = ReferenceManager(populated_dpb, max_refs_l0=5)
    
    # Đang mã hóa frame POC=5
    l0 = rm.build_l0(current_poc=5, slice_type="P")
    
    # Phải lấy các frame quá khứ (4, 2, 0) sắp xếp giảm dần (Gần nhất đứng trước)
    assert l0.pocs == [4, 2, 0], "Lỗi: L0 của P-Slice sắp xếp sai chuẩn!"

def test_build_l0_b_slice(populated_dpb):
    """
    B-slice: L0 ưu tiên khung quá khứ trước, sau đó mới ghép các khung tương lai vào sau.
    """
    rm = ReferenceManager(populated_dpb, max_refs_l0=5)
    
    # Đang mã hóa frame POC=3 (Nằm giữa 2 và 4)
    l0 = rm.build_l0(current_poc=3, slice_type="B")
    
    # Quá khứ xếp giảm dần: [2, 0]. Tương lai xếp tăng dần: [4, 8]
    # Nối lại phải ra: [2, 0, 4, 8]
    assert l0.pocs == [2, 0, 4, 8], "Lỗi: L0 của B-Slice bị nối sai mảng!"

# ---------------------------------------------------------------------------
# 2. TEST XÂY DỰNG L1 (CHỈ DÀNH CHO B-SLICE)
# ---------------------------------------------------------------------------
def test_build_l1_b_slice(populated_dpb):
    """
    B-slice: L1 ưu tiên khung TƯƠNG LAI trước, sau đó mới ghép quá khứ.
    """
    rm = ReferenceManager(populated_dpb, max_refs_l1=5)
    
    # Đang mã hóa frame POC=3
    l1 = rm.build_l1(current_poc=3)
    
    # Tương lai xếp tăng dần: [4, 8]. Quá khứ xếp giảm dần: [2, 0].
    # Nối lại phải ra: [4, 8, 2, 0]
    assert l1.pocs == [4, 8, 2, 0], "Lỗi: L1 của B-Slice không đưa khung tương lai lên đầu!"

# ---------------------------------------------------------------------------
# 3. TEST CƠ CHẾ SINH RPS (CHỮA FATAL BUG DELTA POC)
# ---------------------------------------------------------------------------
def test_rps_negative_delta_poc_sorting(populated_dpb):
    """
    KIỂM CHỨNG LỖI SINH TỬ ĐÃ FIX:
    Mảng delta_poc_neg BẮT BUỘC phải sắp xếp giảm dần để Bitstream không bị vỡ.
    """
    rm = ReferenceManager(populated_dpb)
    
    # Đang mã hóa frame POC=5. Các frame quá khứ trong DPB: 0, 2, 4
    rps = rm.build_rps(current_poc=5, slice_type="P")
    
    # POC hiện tại = 5. Delta của các frame quá khứ là:
    # Frame 4 -> Delta = -1
    # Frame 2 -> Delta = -3
    # Frame 0 -> Delta = -5
    # HEVC yêu cầu mảng này phải là [-1, -3, -5] (Gần nhất đứng trước)
    assert rps.delta_poc_neg == [-1, -3, -5], "FATAL ERROR: Mảng DeltaPocS0 vi phạm chuẩn HEVC!"
    
    # Thử check mảng Positive (Giả sử ta đang đứng ở POC=1, các frame tương lai: 2, 4, 8)
    rps_b = rm.build_rps(current_poc=1, slice_type="B")
    
    # Mảng Positive phải sắp xếp tăng dần: 2-1=1, 4-1=3, 8-1=7 -> [1, 3, 7]
    assert rps_b.delta_poc_pos == [1, 3, 7], "Lỗi: Mảng DeltaPocS1 vi phạm chuẩn!"