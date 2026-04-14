import pytest

from decoded_picture_buffer import (
    DecodedPictureBuffer,
    FrameType,
    RefStatus,
    make_grey_frame
)

# ---------------------------------------------------------------------------
# 1. TEST CƠ CHẾ VÀO/RA CƠ BẢN
# ---------------------------------------------------------------------------
def test_dpb_store_and_fetch():
    """Đảm bảo DPB lưu trữ và truy xuất đúng frame theo POC."""
    dpb = DecodedPictureBuffer(max_buffering=3)
    
    pic = make_grey_frame(poc=0, decode_order=0, frame_type=FrameType.I, width=64, height=64)
    dpb.store(pic)
    
    assert dpb.fullness == 1
    
    fetched = dpb.get_ref(poc=0)
    assert fetched.poc == 0
    assert fetched.frame_type == FrameType.I

def test_dpb_rejects_duplicate_poc():
    """Hệ thống phải báo lỗi nếu lưu 2 frame trùng POC."""
    dpb = DecodedPictureBuffer()
    pic1 = make_grey_frame(poc=5, decode_order=1, frame_type=FrameType.P, width=64, height=64)
    pic2 = make_grey_frame(poc=5, decode_order=2, frame_type=FrameType.P, width=64, height=64)
    
    dpb.store(pic1)
    with pytest.raises(ValueError, match="already exists"):
        dpb.store(pic2)

# ---------------------------------------------------------------------------
# 2. TEST CƠ CHẾ ĐUỔI FRAME (SLIDING WINDOW FIX)
# ---------------------------------------------------------------------------
def test_dpb_evicts_by_decode_order():
    """
    KIỂM CHỨNG BUG ĐÃ FIX:
    Khi DPB đầy, nó phải đuổi frame có decode_order nhỏ nhất, KHÔNG PHẢI poc nhỏ nhất.
    """
    dpb = DecodedPictureBuffer(max_buffering=2)
    
    # Khung P4 vào trước (Decode = 1)
    pic_p4 = make_grey_frame(poc=4, decode_order=1, frame_type=FrameType.P, width=64, height=64)
    dpb.store_and_manage(pic_p4)
    
    # Khung B2 vào sau (Decode = 2)
    pic_b2 = make_grey_frame(poc=2, decode_order=2, frame_type=FrameType.B, width=64, height=64)
    dpb.store_and_manage(pic_b2)
    
    # Nhét thêm khung B3 (Decode = 3). Lúc này DPB (max=2) bị đầy.
    pic_b3 = make_grey_frame(poc=3, decode_order=3, frame_type=FrameType.B, width=64, height=64)
    evicted = dpb.store_and_manage(pic_b3)
    
    # BẮT BUỘC: Frame bị đuổi phải là P4 (Decode = 1, POC = 4).
    # Mặc dù B2 có POC = 2 (nhỏ nhất), nhưng nó mới được đưa vào RAM nên phải được giữ lại.
    assert evicted == [4], f"Lỗi: DPB đã đuổi nhầm frame! Kỳ vọng đuổi POC=4, nhưng lại đuổi {evicted}"
    
    assert 2 in dpb  # POC 2 phải còn sống
    assert 3 in dpb  # POC 3 phải còn sống

# ---------------------------------------------------------------------------
# 3. TEST DANH SÁCH THAM CHIẾU L0 / L1 (L0 / L1 Reference Lists)
# ---------------------------------------------------------------------------
def test_reference_list_sorting():
    """
    Đảm bảo get_preceding_refs (L0) xếp POC giảm dần (gần nhất ở trước).
    Đảm bảo get_following_refs (L1) xếp POC tăng dần.
    """
    dpb = DecodedPictureBuffer(max_buffering=5)
    
    pocs = [0, 2, 4, 8]
    for i, poc in enumerate(pocs):
        dpb.store(make_grey_frame(poc=poc, decode_order=i, frame_type=FrameType.I, width=64, height=64))
        
    # Giả sử ta đang đứng ở POC = 3 (B-frame)
    current_poc = 3
    
    l0_refs = dpb.get_preceding_refs(current_poc)
    l1_refs = dpb.get_following_refs(current_poc)
    
    # L0 (Quá khứ): POC phải nhỏ hơn 3, và xếp giảm dần -> [2, 0]
    assert [p.poc for p in l0_refs] == [2, 0]
    
    # L1 (Tương lai): POC phải lớn hơn 3, và xếp tăng dần -> [4, 8]
    assert [p.poc for p in l1_refs] == [4, 8]

# ---------------------------------------------------------------------------
# 4. TEST IDR FLUSH VÀ BẢO VỆ LONG-TERM
# ---------------------------------------------------------------------------
def test_long_term_protection():
    """Khung Long-term không bao giờ bị cơ chế tự động đuổi."""
    dpb = DecodedPictureBuffer(max_buffering=1)
    
    pic0 = make_grey_frame(poc=0, decode_order=0, frame_type=FrameType.I, width=64, height=64)
    dpb.store(pic0)
    
    # Chuyển thành Long-term
    dpb.mark_long_term(poc=0)
    
    pic1 = make_grey_frame(poc=1, decode_order=1, frame_type=FrameType.P, width=64, height=64)
    
    # Vì frame duy nhất trong buffer là Long-term, cơ chế tự động sẽ bó tay báo lỗi đầy
    with pytest.raises(BufferError, match="full with 1 long-term"):
        dpb.store_and_manage(pic1)

def test_idr_flush():
    """IDR frame phải quét sạch mọi thứ trong DPB."""
    dpb = DecodedPictureBuffer()
    dpb.store(make_grey_frame(poc=0, decode_order=0, frame_type=FrameType.I, width=64, height=64))
    dpb.store(make_grey_frame(poc=1, decode_order=1, frame_type=FrameType.P, width=64, height=64))
    
    assert dpb.fullness == 2
    dpb.flush()
    assert dpb.fullness == 0