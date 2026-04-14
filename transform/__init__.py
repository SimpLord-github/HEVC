"""
transform/__init__.py
Public API cho module Transform & Quantization của HEVC.
"""

# Khai báo các hàm/lớp "Public" (Được phép sử dụng từ bên ngoài)
from .dct import forward_dct, forward_dst
from .quantizer import quantize, dequantize, estimate_rdcost, qp_to_step_size
from .idct import inverse_dct, inverse_dst, full_roundtrip

# Biến __all__ kiểm soát chặt chẽ những gì được export ra ngoài 
# nếu ai đó gõ: `from transform import *`
__all__ = [
    # --- Biến đổi thuận (Forward) ---
    "forward_dct",
    "forward_dst",
    
    # --- Lượng tử hóa (Quantization) ---
    "quantize",
    "dequantize",
    "estimate_rdcost",
    "qp_to_step_size",
    
    # --- Biến đổi ngược (Inverse) ---
    "inverse_dct",
    "inverse_dst",
    
    # --- Tiện ích Test ---
    "full_roundtrip",
]