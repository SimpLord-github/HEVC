"""
ctu_encoder.py — HEVC CTU Encoding Pipeline
Điều phối quá trình mã hóa cho một Coding Tree Unit (CTU) 64x64 đơn lẻ.

Luồng xử lý (Pipeline):
    1. Partitioning & Prediction : Đệ quy chia CTU thành các CU và tìm chế độ dự đoán tốt nhất (Intra/Inter).
    2. Transform & Quantization  : Chia phần dư thành các TU, biến đổi và lượng tử hóa.
    3. Reconstruction            : Giải lượng tử, biến đổi ngược và cộng dự đoán.
    4. Loop Filtering            : Gọi deblock_frame và SAO lên mảng pixel vừa khôi phục.
"""

import numpy as np

# Các cấu trúc dữ liệu lõi
from partitioning.quad_tree import QuadNode, QuadTree, PredMode, CTU_SIZE
from partitioning.cu_split import split_ctu
from partitioning.tu_split import split_tu, reconstruct_cu

# Import CÁC HÀM THỰC TẾ từ Loop Filters của bạn
from loop_filters.deblocking import deblock_frame
from loop_filters.sao import (
    estimate_sao_params, 
    apply_sao_bo, 
    apply_sao_eo, 
    SAOType
)


def encode_ctu(
    frame_luma: np.ndarray,
    ref_frame: np.ndarray | None,
    ctu_x: int,
    ctu_y: int,
    qp: int = 28,
    slice_type: str = "P",
) -> tuple[QuadTree, np.ndarray, list]:
    """
    Thực hiện mã hóa toàn bộ pipeline cho một CTU 64x64.
    """
    
    # =========================================================================
    # 1. KHỞI TẠO VÀ PARTITIONING & PREDICTION (CU SPLIT)
    # =========================================================================
    root = QuadNode(x=ctu_x, y=ctu_y, size=CTU_SIZE, depth=0)
    tree = QuadTree(root=root, frame_w=frame_luma.shape[1], frame_h=frame_luma.shape[0])

    # Chạy thuật toán RDO đệ quy
    split_ctu(root, frame_luma, ref_frame, qp=qp, slice_type=slice_type)

    # Buffer chứa 64x64 pixel khôi phục cho toàn bộ CTU này
    ctu_recon = np.zeros((CTU_SIZE, CTU_SIZE), dtype=np.uint8)
    all_tu_results = []

    # =========================================================================
    # 2. TRANSFORM, QUANTIZATION & RECONSTRUCTION (TU SPLIT)
    # =========================================================================
    for cu_leaf in tree.leaves():
        pm_str = "intra" if cu_leaf.pred_mode == PredMode.INTRA else "inter"
        
        # Biến đổi và lượng tử hóa
        tu_res = split_tu(
            residual=cu_leaf.residual,
            cu_x=cu_leaf.x,
            cu_y=cu_leaf.y,
            cu_size=cu_leaf.size,
            qp=qp,
            pred_mode=pm_str
        )
        all_tu_results.append(tu_res)
        cu_leaf.tu_tree = tu_res
        
        # Khôi phục pixel cho CU này
        cu_recon = reconstruct_cu(
            pred=cu_leaf.pred, 
            tu_tree=tu_res, 
            cu_x=cu_leaf.x, 
            cu_y=cu_leaf.y, 
            cu_size=cu_leaf.size
        )
        
        # Đắp mảnh CU vào CTU lớn
        rx = cu_leaf.x - ctu_x
        ry = cu_leaf.y - ctu_y
        ctu_recon[ry : ry + cu_leaf.size, rx : rx + cu_leaf.size] = cu_recon

    # =========================================================================
    # 3. LOOP FILTERING (DEBLOCKING & SAO)
    # =========================================================================
    
    # 3.1 Deblocking Filter
    # Mảng ctu_recon là 64x64, ta coi nó như một frame luma thu nhỏ.
    # deblock_frame() sẽ tự động quét và làm mượt các ranh giới lưới 8x8 bên trong CTU này (in-place).
    deblock_frame(
        luma=ctu_recon, 
        chroma_cb=None, 
        chroma_cr=None, 
        qp=qp
    )

    # 3.2 Sample Adaptive Offset (SAO)
    # Lấy vùng pixel gốc 64x64 tương ứng để thuật toán SAO so sánh
    H, W = frame_luma.shape
    y_end = min(ctu_y + CTU_SIZE, H)
    x_end = min(ctu_x + CTU_SIZE, W)
    orig_patch = frame_luma[ctu_y:y_end, ctu_x:x_end]

    # Ước lượng tham số SAO (chọn loại BO - Band Offset làm mặc định cho CTU encoder)
    sao_params = estimate_sao_params(
        original=orig_patch, 
        reconstructed=ctu_recon, 
        sao_type=SAOType.BO,   # Có thể thay đổi logic thành quét cả BO và EO để so sánh RD Cost
        x=0, y=0, 
        ctu_size=CTU_SIZE
    )
    
    # Áp dụng bù trừ SAO in-place lên mảng khôi phục
    if sao_params.is_bo:
        apply_sao_bo(plane=ctu_recon, params=sao_params, x=0, y=0, ctu_size=CTU_SIZE)
    elif sao_params.is_eo:
        apply_sao_eo(plane=ctu_recon, params=sao_params, x=0, y=0, ctu_size=CTU_SIZE)

    # Trả về kết quả cuối cùng để CABAC ghi bitstream và DPB lưu làm tham chiếu
    return tree, ctu_recon, all_tu_results