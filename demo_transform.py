# D:\UIT_Doc\Doan1\HEVC\main_encoder.py

from transform import forward_dct, quantize, dequantize, inverse_dct
import numpy as np

if __name__ == "__main__":  
    # Sinh ngẫu nhiên một khối phần dư 8x8
    np.random.seed(12)
    residual = np.random.randint(-100, 100, (8, 8), dtype=np.int16)
    print("Khối phần dư (Residual):")
    print(residual)
    coeffs = forward_dct(residual)
    print("Ma trận Coefficients sau DCT:")
    print(coeffs)
    levels = quantize(coeffs, qp=25)
    print("Ma trận Levels xuất ra từ khối Transform:")
    print(levels)
    dequantized = dequantize(levels, qp=25)
    reconstructed = inverse_dct(dequantized)
    print("Khối tái tạo sau khi dequantize và inverse DCT:")
    print(reconstructed)
    