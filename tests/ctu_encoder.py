"""
ctu_encoder.py — HEVC CTU Pipeline Master
Điều phối quá trình nén hoàn chỉnh cho một khối CTU 64x64.
"""

import numpy as np

# Import từ các thư mục cốt lõi của kiến trúc
from quad_tree import QuadNode, QuadTree, PredMode, CTU_SIZE
from cu_split import split_ctu
from tu_split import split_tu, reconstruct_cu

# ---------------------------------------------------------------------------
# PUBLIC API: ENCODE CTU
# ---------------------------------------------------------------------------
def encode_ctu(
    frame_luma: np.ndarray,
    ref_frame:  np.ndarray | None,
    ctu_x:      int,
    ctu_y:      int,
    qp:         int = 28,
    slice_type: str = "P",
) -> tuple[QuadTree, np.ndarray, list]:
    """
    Nén một khối CTU 64x64 hoàn chỉnh (Bao gồm CU Split + TU Split + Reconstruct).
    
    Returns:
        - Cây CU (QuadTree) đã được quyết định toàn bộ chế độ.
        - Mảng Reconstructed Pixel của CTU này (64x64 uint8).
        - Danh sách các Cây TU (TUResult) tương ứng với từng lá CU.
    """
    # 1. Khởi tạo Gốc CTU
    root = QuadNode(x=ctu_x, y=ctu_y, size=CTU_SIZE, depth=0)
    tree = QuadTree(root=root, frame_w=frame_luma.shape[1], frame_h=frame_luma.shape[0])

    # 2. CHẠY NÃO BỘ DỰ ĐOÁN (CU Split Decision & Mode Decision)
    # Hàm này sẽ tự động đệ quy và chia nhỏ 64x64 xuống các khối nhỏ hơn nếu cần.
    split_ctu(root, frame_luma, ref_frame, qp=qp, slice_type=slice_type)

    # 3. CHẠY NÃO BỘ BIẾN ĐỔI (TU Split) VÀ KHÔI PHỤC (Reconstruction)
    ctu_recon = np.zeros((CTU_SIZE, CTU_SIZE), dtype=np.uint8)
    all_tu_results = []

    for cu_leaf in tree.leaves():
        # Chuyển đổi định dạng Enum PredMode sang string ("intra" / "inter")
        pm_str = "intra" if cu_leaf.pred_mode == PredMode.INTRA else "inter"
        
        # Gọi RQT (Residual Quad-Tree) để chia nhỏ phần dư của lá CU này
        tu_res = split_tu(
            residual=cu_leaf.residual,
            cu_x=cu_leaf.x,
            cu_y=cu_leaf.y,
            cu_size=cu_leaf.size,
            qp=qp,
            pred_mode=pm_str
        )
        all_tu_results.append(tu_res)
        
        # Gắn TU result vào CU node để tiện theo dõi
        cu_leaf.tu_tree = tu_res
        
        # Đắp phần dư (sau giải mã) vào ảnh dự đoán để tạo ra ảnh khôi phục (Reconstructed)
        cu_recon = reconstruct_cu(cu_leaf.pred, tu_res, cu_leaf.x, cu_leaf.y, cu_leaf.size)
        
        # Ghép mảnh CU khôi phục này vào bức tranh CTU 64x64 tổng thể
        rx = cu_leaf.x - ctu_x
        ry = cu_leaf.y - ctu_y
        ctu_recon[ry : ry + cu_leaf.size, rx : rx + cu_leaf.size] = cu_recon

    return tree, ctu_recon, all_tu_results

# ===========================================================================
# 🚀 KỊCH BẢN CHẠY DEMO (SAMPLE INPUT) 
# ===========================================================================
if __name__ == "__main__":
    import sys
    import os
    from unittest.mock import MagicMock

    # 1. Tiêm đường dẫn để Python hiểu cấu trúc dự án khi chạy trực tiếp file này
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    # 2. MOCKING TẦNG TRANSFORM 
    sys.modules['transform'] = MagicMock()
    mock_dct = MagicMock()
    mock_dct.forward_dct.side_effect = lambda x: x.astype(np.int32)
    mock_dct.forward_dst.side_effect = lambda x: x.astype(np.int32)
    sys.modules['transform.dct'] = mock_dct

    mock_q = MagicMock()
    mock_q.quantize.side_effect = lambda c, qp, is_intra: np.where(np.abs(c) > 10, c, 0)
    mock_q.dequantize.side_effect = lambda l, qp: l
    sys.modules['transform.quantizer'] = mock_q

    mock_idct = MagicMock()
    mock_idct.inverse_dct.side_effect = lambda x: x.astype(np.int16)
    mock_idct.inverse_dst.side_effect = lambda x: x.astype(np.int16)
    sys.modules['transform.idct'] = mock_idct

    # 3. TẠO SAMPLE INPUT (Một khung hình 64x64 với độ phức tạp cao)
    print("==========================================================")
    print("      🚀 HEVC CTU ENCODER PIPELINE - SAMPLE INPUT 🚀      ")
    print("==========================================================\n")
    
    print("[1] Khởi tạo dữ liệu ảnh thô (Raw Frame)...")
    np.random.seed(42)
    
    # Tạo một khung ảnh nền màu xám 128
    original_frame = np.full((64, 64), 128, dtype=np.uint8)
    
    # BẪY RDO: Trộn một dải nhiễu loạn cực mạnh vào góc dưới bên phải (SE)
    # Điều này sẽ ép bộ não Cây Không Gian (Partitioning) phải băm nát góc này ra!
    original_frame[32:64, 32:64] = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    
    # Khung tham chiếu Inter (trống) -> Ép hệ thống dùng slice_type="I" (Intra)
    ref_frame = None  

    # 4. CHẠY PIPELINE
    print("[2] Khởi động động cơ CTU Encoder...")
    print("    -> Đang chạy CU Quad-Tree Search (Tìm dự đoán tối ưu)...")
    print("    -> Đang chạy TU Quad-Tree Search (Tìm biến đổi dư tối ưu)...")
    
    tree, recon_ctu, all_tu = encode_ctu(
        frame_luma=original_frame,
        ref_frame=ref_frame,
        ctu_x=0, ctu_y=0,
        qp=28,
        slice_type="I"
    )

    # 5. HIỂN THỊ KẾT QUẢ ĐẦU RA
    print("\n==========================================================")
    print("               🏆 KẾT QUẢ MÃ HÓA CTU 64x64 🏆             ")
    print("==========================================================")
    
    total_sse = sum(tu.total_sse for tu in all_tu)
    total_nnz = sum(tu.total_nnz for tu in all_tu)
    
    print(f"Tổng số lá CU (Coding Units)   : {tree.count_leaves()} khối")
    print(f"Tổng Méo dạng (CTU SSE)        : {total_sse}")
    print(f"Tổng Lượng Bit Xấp xỉ (NNZ)    : {total_nnz} hệ số khác 0\n")

    print("CẤU TRÚC PHÂN MẢNH CÂY CU & TU (Quad-Tree Hierarchy):")
    print("----------------------------------------------------------")
    for i, cu in enumerate(tree.leaves(), 1):
        tu_leaves_count = len(cu.tu_tree.leaves)
        print(f"CU {i:02d} | Tọa độ ({cu.x:02d}, {cu.y:02d}) | Size: {cu.size:02d}x{cu.size:02d} | Chế độ: {cu.pred_mode.name:5s} | Chứa {tu_leaves_count} lá TU")
        
        # Nếu TU bị chẻ (Số lá TU > 1), in chi tiết ra
        if tu_leaves_count > 1:
            for j, tu in enumerate(cu.tu_tree.leaves, 1):
                print(f"       -> TU {j} | Size: {tu.size:02d}x{tu.size:02d} | Cost: {tu.rd_cost:.1f}")
    
    print("\n[Trích xuất 8x8 Pixel Khôi Phục (Reconstructed) ở góc nhiễu]")
    print(recon_ctu[32:40, 32:40])
    print("==========================================================")