import numpy as np
from prediction import decide_mode

def run_prediction_demo():
    # Cố định random seed để kết quả luôn giống nhau mỗi khi bạn chạy
    np.random.seed(42)

    n = 8
    origin = (16, 16)

    # ---------------------------------------------------------
    # BƯỚC 1: KHỞI TẠO MA TRẬN ĐẦU VÀO (RANDOM)
    # ---------------------------------------------------------
    print("[1] Đang sinh dữ liệu đầu vào (Input Generation)...")
    
    # Block hiện tại cần nén (8x8)
    block = np.random.randint(50, 200, (n, n), dtype=np.uint8)
    
    # Viền tham chiếu cho dự đoán Intra (màu ngẫu nhiên)
    ref_above = np.random.randint(50, 200, 2*n + 1, dtype=np.int16)
    ref_left  = np.random.randint(50, 200, 2*n + 1, dtype=np.int16)
    
    # Khung tham chiếu cho dự đoán Inter (64x64)
    ref_frame = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    
    # [BẪY INTER]: Copy block hiện tại và giấu vào ref_frame ở vị trí dịch chuyển (dx=2, dy=-3)
    ref_frame[origin[1]-3 : origin[1]-3+n, origin[0]+2 : origin[0]+2+n] = block.copy()

    print(f"  - Current Block    : {n}x{n} pixels")
    print(f"  - Intra Refs       : Chiều dài {2*n + 1}")
    print(f"  - Ref Frame (Inter): 64x64 pixels")
    print(f"  - Tọa độ gốc       : x={origin[0]}, y={origin[1]}\n")

    # ---------------------------------------------------------
    # BƯỚC 2: CHẠY QUA BỘ RDO ARBITER
    # ---------------------------------------------------------
    print("[2] Đang qua RDO (Mode Decision)...")
    qp = 28
    
    # Gọi hàm lõi của hệ thống Prediction
    decision = decide_mode(
        block=block,
        ref_above=ref_above,
        ref_left=ref_left,
        ref_frame=ref_frame,
        origin=origin,
        qp=qp,
        slice_type="P",
        search_range=16
    )

    
    print(f"Chế độ chiến thắng (Winning Mode) : [ {decision.mode.upper()} ]")
    print(f"Tổng RD Cost (J = D + lambda*R)   : {decision.rd_cost:.2f}")
    print(f"Méo dạng (SSE Distortion - D)     : {decision.distortion}")
    print(f"Chi phí Rate Proxy (R)            : {decision.rate_proxy} bits")
    print(f"Hệ số Lagrange (Lambda tại QP=28) : {decision.lambda_:.2f}")

    if decision.is_inter:
        # Nhắc lại: MV lưu trữ dạng Quarter-pel, nên chia 4 để xem Pixel thực
        print(f"Vector chuyển động (MV)           : dx = {decision.mv.mvx_int}, dy = {decision.mv.mvy_int} (Pixel)")
    elif decision.is_intra:
        print(f"Chế độ Intra (Intra Mode)         : Mode {decision.intra_mode}")

    print("\n[Trích xuất 4x4 góc trái trên của Block Gốc]")
    print(block[:4, :4])

    print("\n[Trích xuất 4x4 góc trái trên của Ảnh Dự Đoán - Prediction]")
    print(decision.pred[:4, :4])

    print("\n[Trích xuất 4x4 góc trái trên của Phần Dư - Residual]")
    print(decision.residual[:4, :4])
    print("==========================================================")

if __name__ == "__main__":
    run_prediction_demo()