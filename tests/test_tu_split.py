import sys
import pytest
import numpy as np
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# MOCKING TẦNG TRANSFORM (Để test không bị crash do thiếu module)
# ---------------------------------------------------------------------------
sys.modules['transform'] = MagicMock()

# Mock DCT/DST: Trả về y nguyên giá trị đầu vào để pass mượt mà
mock_dct = MagicMock()
mock_dct.forward_dct.side_effect = lambda x: x.astype(np.int32)
mock_dct.forward_dst.side_effect = lambda x: x.astype(np.int32)
sys.modules['transform.dct'] = mock_dct

# Mock Quantizer: Cố tình "xóa" các số < 10 thành 0 để tạo ra nnz (rate proxy)
mock_q = MagicMock()
mock_q.quantize.side_effect = lambda c, qp, is_intra: np.where(np.abs(c) > 10, c, 0)
mock_q.dequantize.side_effect = lambda l, qp: l
sys.modules['transform.quantizer'] = mock_q

# Mock Inverse: Trả lại định dạng cũ
mock_idct = MagicMock()
mock_idct.inverse_dct.side_effect = lambda x: x.astype(np.int16)
mock_idct.inverse_dst.side_effect = lambda x: x.astype(np.int16)
sys.modules['transform.idct'] = mock_idct

# BÂY GIỜ MỚI IMPORT MODULE CẦN TEST
from tu_split import split_tu
from quad_tree import NodeState

# ---------------------------------------------------------------------------
# BÀI TEST CHÍNH THỨC
# ---------------------------------------------------------------------------

def test_intra_8x8_split_constraint():
    """
    Theo chuẩn HEVC, Intra 8x8 CU KHÔNG ĐƯỢC phép chẻ nhỏ Residual thành 4x4.
    Cây bắt buộc phải chốt ở LEAF 8x8.
    """
    res = np.random.randint(-50, 50, (8, 8), dtype=np.int16)
    
    result = split_tu(res, cu_x=0, cu_y=0, cu_size=8, pred_mode="intra")
    
    assert result.root.state == NodeState.LEAF, "Lỗi: Khối Intra 8x8 đã bị chẻ nhỏ sai chuẩn!"
    assert result.root.size == 8
    assert len(result.leaves) == 1

def test_inter_16x16_split_decision():
    """
    Inter CU thì được phép chẻ tự do.
    Ta tạo một residual 16x16 phẳng lì (không lỗi), nhưng cấy một cục lỗi khổng lồ
    ở góc NW (Trái Trên). RDO chắc chắn sẽ phải chẻ nhỏ nó ra để cô lập vùng nhiễu.
    """
    res = np.zeros((16, 16), dtype=np.int16)
    
    # Cấy cục lỗi khổng lồ (sẽ tạo ra nhiều hệ số non-zero) vào góc NW
    res[0:8, 0:8] = 200 
    
    result = split_tu(res, cu_x=0, cu_y=0, cu_size=16, pred_mode="inter")
    
    # Do vùng lỗi dồn hết vào 1 góc, RDO sẽ tách TU ra để tiết kiệm bit ở 3 góc còn lại
    assert result.root.state == NodeState.SPLIT, "Lỗi: Residual không đồng đều thì RDO phải chẻ ra."
    assert len(result.leaves) >= 4

def test_reconstruct_tu_pipeline():
    """Đảm bảo hàm reconstruct trả về đủ block pixel."""
    from tu_split import reconstruct_cu
    res = np.zeros((16, 16), dtype=np.int16)
    pred = np.full((16, 16), 128, dtype=np.uint8)
    
    # Ép một cây TU chạy ra
    tu_result = split_tu(res, cu_x=10, cu_y=10, cu_size=16, pred_mode="inter")
    
    recon = reconstruct_cu(pred, tu_result, cu_x=10, cu_y=10, cu_size=16)
    
    assert recon.shape == (16, 16)
    assert recon.dtype == np.uint8