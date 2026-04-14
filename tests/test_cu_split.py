import pytest
import numpy as np

from quad_tree import QuadNode, NodeState
from cu_split import split_ctu

# ---------------------------------------------------------------------------
# 1. TEST BẮT BUỘC CHẺ CTU 64x64
# ---------------------------------------------------------------------------
def test_forced_ctu_split():
    """Chuẩn HEVC quy định CTU 64x64 bắt buộc phải chẻ xuống ít nhất 32x32."""
    frame = np.zeros((64, 64), dtype=np.uint8)
    root = QuadNode(0, 0, 64, 0)
    
    split_ctu(root, frame_luma=frame, ref_frame=None, qp=28, slice_type="I")
    
    assert root.state == NodeState.SPLIT, "Lỗi: Khối 64x64 bắt buộc phải SPLIT."
    assert len(root.children) == 4

# ---------------------------------------------------------------------------
# 2. TEST ẢNH PHẲNG (LEAF Thắng)
# ---------------------------------------------------------------------------
def test_leaf_wins_on_flat_image():
    """
    Tạo một bức ảnh phẳng lỳ (màu 128). Khối 32x32 dùng mode DC/Planar bao phủ hoàn hảo.
    Chẻ nhỏ chỉ tốn thêm bits. RDO phải chốt giữ nguyên 32x32 (LEAF).
    """
    frame = np.full((64, 64), 128, dtype=np.uint8)
    root = QuadNode(16, 16, 32, 1)  # Khởi đầu ở 32x32 để tránh bị ép luật chẻ 64
    
    split_ctu(root, frame_luma=frame, ref_frame=None, qp=28, slice_type="I")
    
    assert root.state == NodeState.LEAF, "Lỗi: Ảnh phẳng lỳ thì RDO không được chẻ!"
    assert root.size == 32

# ---------------------------------------------------------------------------
# 3. TEST ẢNH PHỨC TẠP (Bẫy Split thông minh)
# ---------------------------------------------------------------------------
def test_split_wins_on_complex_image():
    """
    Bẫy SPLIT dựa trên đặc tính Feed-forward Intra Reference của Golden Model.
    Khung nền màu 128, ở giữa cấy một khối 32x32 màu 200.
    - LEAF 32x32: Viền ngoài là 128 -> Dự đoán ra 128 so với gốc 200 -> SSE khổng lồ ~5.3M.
    - SPLIT ra 16x16: RDO sẽ tính toán chẻ nhỏ ra. Thậm chí các khối 16x16 cũng sẽ 
      tự động chẻ tiếp xuống 8x8 để "lẩn tránh" các viền tham chiếu xấu (128) 
      và tối đa hóa các khối đạt SSE = 0.
    """
    frame = np.full((64, 64), 128, dtype=np.uint8)
    frame[16:48, 16:48] = 200
    
    root = QuadNode(16, 16, 32, 1)
    
    split_ctu(root, frame_luma=frame, ref_frame=None, qp=28, slice_type="I")
    
    # Mục tiêu tối thượng của bài test: Phải chẻ cái node Root ra!
    assert root.state == NodeState.SPLIT, "Lỗi: J_split rẻ hơn mà RDO lại không chịu chẻ!"
    
    # Chứng minh RDO đệ quy hoạt động: Cây phải có độ sâu > 1 (tức là đã chẻ ít nhất 1 lần)
    def max_depth(node):
        if not node.children:
            return node.depth
        return max(max_depth(c) for c in node.children)
        
    assert max_depth(root) > 1, "Cây chưa hề được đệ quy chẻ sâu xuống!"