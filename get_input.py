import cv2
import numpy as np

def extract_yuv_matrix_for_hevc(image_path, output_yuv_path):
    """
    Đọc ảnh, chuyển đổi sang ma trận điểm ảnh YUV420p và xuất ra file nhị phân cho HEVC.
    """
    # 1. Đọc ảnh bằng OpenCV (mặc định OpenCV đọc ảnh dưới dạng BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc được ảnh từ đường dẫn: {image_path}")

    height, width = img.shape[:2]
    print(f"Kích thước ảnh gốc: {width}x{height}")

    # 2. Đảm bảo kích thước ảnh là số chẵn
    # Yêu cầu bắt buộc của Chroma Subsampling 4:2:0 là kích thước phải chia hết cho 2
    if width % 2 != 0 or height % 2 != 0:
        print("Cảnh báo: Cắt/Resize ảnh về kích thước chẵn để phù hợp với định dạng YUV420p.")
        width = width - (width % 2)
        height = height - (height % 2)
        img = cv2.resize(img, (width, height))

    # 3. Chuyển đổi sang YUV420p (I420)
    # cv2.COLOR_BGR2YUV_I420 sẽ tạo ra một ma trận 1D (planar) chứa:
    # - Plane Y: kích thước (width x height)
    # - Plane U: kích thước (width/2 x height/2)
    # - Plane V: kích thước (width/2 x height/2)
    yuv_matrix = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)

    # 4. Ghi ma trận điểm ảnh ra file nhị phân thô (raw binary)
    with open(output_yuv_path, "wb") as f:
        f.write(yuv_matrix.tobytes())

    print(f"Thành công! Đã xuất ma trận điểm ảnh YUV ra: {output_yuv_path}")
    print(f"Lưu ý: Khi đưa vào HEVC encoder, hãy khai báo độ phân giải là {width}x{height} và format là yuv420p.")
    
    return yuv_matrix

# --- Thực thi ---
if __name__ == "__main__":
    input_image = "hinh.jfif"   # Đổi thành tên ảnh của bạn
    output_raw = "output.yuv"   # File đầu vào cho HEVC
    # 1. Read the raw binary data
with open("output.yuv", "rb") as f:
    raw_bytes = f.read()
width = 1920  # Replace with your image's width
height = 1080 # Replace with your image's height
# 2. Convert bytes to a 1D numpy array
# The total size of YUV420p is width * height * 1.5 bytes
yuv_array = np.frombuffer(raw_bytes, dtype=np.uint8)

# 3. Reshape the 1D array into the YUV planar matrix structure
yuv_matrix = yuv_array.reshape((int(height * 1.5), width))

# 4. Convert back to BGR so your screen can display it properly
bgr_img = cv2.cvtColor(yuv_matrix, cv2.COLOR_YUV2BGR_I420)
try:
    ma_tran = extract_yuv_matrix_for_hevc(input_image, output_raw)
        # Biến ma_tran hiện tại chứa mảng numpy của YUV420p nếu bạn muốn xử lý thêm trong RAM
except Exception as e:
    print(f"Lỗi: {e}")
# 5. Display the matrix
cv2.imshow("The Matrix (YUV Decoded)", bgr_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
   