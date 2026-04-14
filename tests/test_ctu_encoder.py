"""
test_ctu_encoder.py — Unit tests cho phiên bản cập nhật của CTU Encoder
Kiểm tra luồng xử lý: CTU -> CUs -> TUs -> Reconstruction -> Deblocking -> SAO
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Import hàm cần test và cấu trúc dữ liệu
from ctu_encoder import encode_ctu
from quad_tree import QuadNode, PredMode, CTU_SIZE

class TestCTUEncoderPipeline(unittest.TestCase):

    # Patch các hàm thực tế đã được import bên trong pipeline.ctu_encoder
    @patch('ctu_encoder.apply_sao_eo')
    @patch('ctu_encoder.apply_sao_bo')
    @patch('ctu_encoder.estimate_sao_params')
    @patch('ctu_encoder.deblock_frame')
    @patch('ctu_encoder.reconstruct_cu')
    @patch('ctu_encoder.split_tu')
    @patch('ctu_encoder.split_ctu')
    def test_encode_ctu_pipeline_flow(
        self, 
        mock_split_ctu, 
        mock_split_tu, 
        mock_reconstruct_cu, 
        mock_deblock_frame, 
        mock_estimate_sao_params,
        mock_apply_sao_bo,
        mock_apply_sao_eo
    ):
        # =====================================================================
        # 1. SETUP DỮ LIỆU ĐẦU VÀO
        # =====================================================================
        frame_w, frame_h = 128, 128
        frame_luma = np.full((frame_h, frame_w), 128, dtype=np.uint8)
        ref_frame = None  # Giả lập I-Slice
        ctu_x, ctu_y = 0, 0
        qp = 28

        # =====================================================================
        # 2. GIẢ LẬP (MOCK) CÁC HÀM CON
        # =====================================================================
        
        # Giả lập hàm split_ctu: Tự động chia CTU 64x64 thành 4 khối CU 32x32
        def side_effect_split_ctu(root_node, *args, **kwargs):
            children = root_node.split() # Chia làm 4 (NW, NE, SW, SE)
            for child in children:
                child.mark_leaf()
                child.pred_mode = PredMode.INTRA
                child.pred = np.zeros((32, 32), dtype=np.uint8)
                child.residual = np.zeros((32, 32), dtype=np.int16)
        mock_split_ctu.side_effect = side_effect_split_ctu

        # Giả lập hàm split_tu: Trả về một Mock object
        mock_tu_result = MagicMock()
        mock_split_tu.return_value = mock_tu_result

        # Giả lập hàm reconstruct_cu: Trả về một mảng 32x32 có giá trị = 200
        mock_reconstruct_cu.return_value = np.full((32, 32), 200, dtype=np.uint8)

        # Giả lập SAO Params: Ép nó trả về loại BO (Band Offset)
        mock_sao_params = MagicMock()
        mock_sao_params.is_bo = True
        mock_sao_params.is_eo = False
        mock_estimate_sao_params.return_value = mock_sao_params

        # =====================================================================
        # 3. THỰC THI HÀM (EXECUTE)
        # =====================================================================
        tree, ctu_recon, all_tu_results = encode_ctu(
            frame_luma=frame_luma,
            ref_frame=ref_frame,
            ctu_x=ctu_x,
            ctu_y=ctu_y,
            qp=qp,
            slice_type="I"
        )

        # =====================================================================
        # 4. KIỂM TRA KẾT QUẢ (ASSERTIONS)
        # =====================================================================
        
        # A. Kiểm tra Cây CU
        self.assertEqual(tree.count_leaves(), 4, "CTU phải được chia thành 4 lá CU")
        mock_split_ctu.assert_called_once()

        # B. Kiểm tra quá trình TU & Khôi phục (Reconstruction)
        self.assertEqual(mock_split_tu.call_count, 4, "Phải gọi split_tu 4 lần")
        self.assertEqual(mock_reconstruct_cu.call_count, 4, "Phải gọi reconstruct_cu 4 lần")
        
        # Mảng 64x64 khôi phục phải lấp đầy số 200 (do ghép 4 khối 32x32 lại)
        self.assertEqual(ctu_recon.shape, (64, 64))
        self.assertTrue(np.all(ctu_recon == 200), "Các mảnh CU chưa được đắp đúng vị trí")

        # C. Kiểm tra Loop Filters (Deblocking & SAO)
        mock_deblock_frame.assert_called_once()
        
        # Đảm bảo estimate_sao_params được gọi với đúng kích thước CTU
        mock_estimate_sao_params.assert_called_once()
        
        # Do ta đã mock sao_params.is_bo = True, hàm apply_sao_bo phải được gọi
        mock_apply_sao_bo.assert_called_once()
        # Hàm apply_sao_eo không được phép gọi
        mock_apply_sao_eo.assert_not_called()

if __name__ == '__main__':
    unittest.main()