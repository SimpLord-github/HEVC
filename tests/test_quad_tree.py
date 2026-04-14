import pytest
from quad_tree import (
    QuadNode,
    QuadTree,
    NodeState,
    build_flat_tree,
    build_full_tree,
    build_uniform_tree,
    CTU_SIZE,
    MIN_CU_SIZE
)

# ---------------------------------------------------------------------------
# 1. TEST CÁC THAO TÁC CƠ BẢN TRÊN NODE (Node Operations)
# ---------------------------------------------------------------------------
def test_node_split_geometry():
    """Kiểm tra xem khi split, 4 node con có được chia tọa độ và kích thước đúng chuẩn không."""
    node = QuadNode(x=10, y=20, size=32, depth=1)
    
    # Chưa split
    assert node.state == NodeState.UNSPLIT
    assert node.is_leaf is False
    
    children = node.split()
    
    assert node.state == NodeState.SPLIT
    assert len(children) == 4
    
    nw, ne, sw, se = children
    
    # Kích thước phải giảm một nửa
    assert nw.size == 16 and ne.size == 16 and sw.size == 16 and se.size == 16
    
    # Tọa độ (Raster order: Trái-Trên, Phải-Trên, Trái-Dưới, Phải-Dưới)
    assert nw.origin == (10, 20)
    assert ne.origin == (26, 20)   # x + 16
    assert sw.origin == (10, 36)   # y + 16
    assert se.origin == (26, 36)   # x + 16, y + 16

def test_node_split_exceptions():
    """Đảm bảo hệ thống chặn các hành vi xẻ cây sai luật."""
    node = QuadNode(x=0, y=0, size=MIN_CU_SIZE, depth=4)
    
    # Không được xẻ nhỏ hơn MIN_CU_SIZE
    with pytest.raises(ValueError, match="already at minimum"):
        node.split()
        
    valid_node = QuadNode(x=0, y=0, size=16, depth=2)
    valid_node.mark_leaf()
    
    # Node đã chốt là LEAF thì không được split nữa
    with pytest.raises(ValueError, match="cannot be split again"):
        valid_node.split()

# ---------------------------------------------------------------------------
# 2. TEST CÁC BỘ XÂY DỰNG CÂY (Tree Builders)
# ---------------------------------------------------------------------------
def test_build_flat_tree():
    """Tree phẳng (không xẻ): Chỉ có 1 node gốc làm LEAF."""
    tree = build_flat_tree(ctu_x=0, ctu_y=0, ctu_size=64)
    
    assert tree.count_nodes() == 1
    assert tree.count_leaves() == 1
    assert tree.max_depth() == 0
    assert tree.root.is_leaf is True
    assert tree.total_area() == 64 * 64

def test_build_uniform_tree():
    """Tree xẻ đều xuống kích thước 16x16. Từ 64x64 sẽ ra 16 blocks."""
    tree = build_uniform_tree(ctu_x=0, ctu_y=0, cu_size=16, ctu_size=64)
    
    # CTU 64x64 -> 4 node 32x32 -> 16 node 16x16
    # Tổng số LEAF = 16
    assert tree.count_leaves() == 16
    
    # Mọi lá đều phải có size 16
    sizes = tree.leaf_sizes()
    assert len(sizes) == 1
    assert sizes.get(16) == 16
    
    # Độ sâu tối đa: 64(0) -> 32(1) -> 16(2)
    assert tree.max_depth() == 2
    assert tree.total_area() == 64 * 64

def test_build_full_tree():
    """Tree xẻ nát bét xuống tận MIN_CU_SIZE (4x4)."""
    tree = build_full_tree(ctu_x=0, ctu_y=0, min_size=4, ctu_size=64)
    
    # CTU 64x64, xẻ thành các block 4x4 -> Tổng cộng có (64/4)*(64/4) = 16*16 = 256 leaves
    assert tree.count_leaves() == 256
    assert tree.max_depth() == 4
    assert tree.total_area() == 64 * 64

# ---------------------------------------------------------------------------
# 3. TEST TÍNH TOÀN VẸN CỦA TREE (Validation Logic)
# ---------------------------------------------------------------------------
def test_tree_validation_success():
    """Cây chuẩn phải đi qua validate() mà không có lỗi."""
    tree = build_uniform_tree(0, 0, cu_size=8)
    errors = tree.validate()
    assert len(errors) == 0, f"Cây chuẩn nhưng lại sinh lỗi: {errors}"

def test_tree_validation_catches_corruption():
    """Cố tình phá hỏng cấu trúc cây để xem hàm validate có phát hiện được không."""
    tree = build_uniform_tree(0, 0, cu_size=32)
    
    # Phá hoại 1: Đổi kích thước của một node con (làm sai lệch diện tích)
    first_leaf = next(tree.leaves())
    first_leaf.size = 10 
    
    errors = tree.validate()
    assert len(errors) > 0
    # Phải có lỗi diện tích
    assert any("Area mismatch" in err for err in errors)
    # Phải có lỗi sai geometry so với thuật toán xẻ
    assert any("Child geometry wrong" in err for err in errors)