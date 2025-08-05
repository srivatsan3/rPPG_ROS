import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from time import perf_counter
import cv2
import mediapipe as mp
import numpy as np
import torch
import sys
import os
import psutil
rppg_tb_path = '/home/mscrobotics2425laptop11/Dissertation/rppgtb/rPPG-Toolbox'

sys.path.insert(0, rppg_tb_path) # Path to rPPG toolbox

# Importing different Neural Network methods from the rPPG Toolbox
from neural_methods.model.EfficientPhys import EfficientPhys
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.DeepPhys import DeepPhys
from neural_methods.model.BigSmall import BigSmall

from evaluation.post_process import _calculate_peak_hr, _calculate_fft_hr
    # Landmark indices for approximate ROIs
FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251]
LEFT_CHEEK_LANDMARKS = [234, 93, 132]
RIGHT_CHEEK_LANDMARKS = [454, 323, 361]
    

class RPPGNeuralNode(Node):
    def __init__(self):
        super().__init__('rppg_neural_node')
        self.publisher_ = self.create_publisher(Float32, 'heart_rate_bpm', 10)
        self.cap = cv2.VideoCapture(0)

        self.fps = 30                # Video Frames per second
        self.window_size = 240       # Frame Size (fps * number of seconds)
        self.roi_size = 72           # Region of Interest crop size
        self.buffer = []             # Empty buffer
        self.model_name = 'PhysNet' # Model to choose

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, checkpoint_path = self.load_model(model = self.model_name, frames = self.window_size, img_size = self.roi_size)
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned_state)
        self.model.eval()

        # MediaPipe face detection
        self.face_detector = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

        self.mp_face_mesh = mp.solutions.face_mesh


    def get_memory_usage_mb(self):
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss  # Resident Set Size in bytes
        return mem_bytes / (1024 ** 2)  # Convert to MB

    
    def load_model(self,model,frames = 300, img_size= 72):
        if model == 'EfficientPhys':
            self.model = EfficientPhys(frame_depth=frames, img_size=img_size).to(self.device)
            checkpoint_path = rppg_tb_path+'/final_model_release/UBFC-rPPG_EfficientPhys.pth'
        if model == 'PhysNet':
            self.model = PhysNet_padding_Encoder_Decoder_MAX(frames=frames).to(self.device)
            checkpoint_path = rppg_tb_path+'/final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth'
        if model == 'BigSmall':
            self.model = BigSmall().to(self.device)
            checkpoint_path = rppg_tb_path+'/final_model_release/BP4D_BigSmall_Multitask_Fold1.pth'
        if model == 'DeepPhys':
            self.model = DeepPhys(img_size = img_size).to(self.device)
            checkpoint_path = rppg_tb_path+'final_model_release/UBFC-rPPG_DeepPhys.pth'

        return self.model, checkpoint_path


    def extract_face_crop(self, frame):
        results = self.face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return None, None

        detection = results.detections[0]
        h, w, _ = frame.shape
        bbox = detection.location_data.relative_bounding_box

        x1 = int(bbox.xmin * w)
        y1 = int(bbox.ymin * h)
        x2 = int((bbox.xmin + bbox.width) * w)
        y2 = int((bbox.ymin + bbox.height) * h)

        if y2 <= y1 or x2 <= x1:
            return None, None  # Invalid crop

        face = frame[y1:y2, x1:x2]

        if face is None or face.size == 0:
            return None, None  # Empty crop
        face_resized = cv2.resize(face, (self.roi_size, self.roi_size))
        forehead_h_ratio = 0.3
        forehead_y2 = y1 + int((y2 - y1) * forehead_h_ratio)
        forehead = frame[y1:forehead_y2, x1:x2]
        forehead_resized = cv2.resize(forehead, (self.roi_size, self.roi_size)) if forehead.size != 0 else None


        return forehead_resized, detection



    def extract_face_regions(self,frame, roi_size=128):
        with self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                    refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None

            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark

            def extract_roi(landmark_indices):
                points = [landmarks[i] for i in landmark_indices]
                xs = [int(p.x * w) for p in points]
                ys = [int(p.y * h) for p in points]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    return None
                return cv2.resize(roi, (roi_size, roi_size))

            # return {
            #     "forehead": extract_roi(FOREHEAD_LANDMARKS),
            #     "left_cheek": extract_roi(LEFT_CHEEK_LANDMARKS),
            #     "right_cheek": extract_roi(RIGHT_CHEEK_LANDMARKS)
            # }
            return extract_roi(FOREHEAD_LANDMARKS), landmarks


    def prepare_input_for_bigsmall(self, big_res=144, small_res=9):

        raw_np = np.array(self.buffer).astype(np.float32) / 255.0  # [T, H, W, C]
        T, H, W, C = raw_np.shape


        big_np = np.stack([cv2.resize(f, (big_res, big_res)) for f in raw_np])  # [T, H_big, W_big, C]

        diff_np = np.diff(big_np, axis=0)  # [T-1, H_big, W_big, C]
        diff_np = np.pad(diff_np, ((1,0),(0,0),(0,0),(0,0)), mode='constant')  # Pad to match length
        diff_np = np.clip(diff_np * 5.0, -1.0, 1.0)  # Rescale motion amplitude

        small_np = np.stack([cv2.resize(f, (small_res, small_res)) for f in diff_np])  # [T, H_small, W_small, C]

        big_tensor   = torch.tensor(big_np).permute(0, 3, 1, 2).to(self.device)    # [T, C, H_big, W_big]
        small_tensor = torch.tensor(small_np).permute(0, 3, 1, 2).to(self.device)  # [T, C, H_small, W_small]

        return [big_tensor, small_tensor]
    
    def prepare_input_for_deepphys(self):
        raw_np = np.array(self.buffer) / 255.0  # [T, H, W, C]
        diff_np = np.diff(raw_np, axis=0)       # [T-1, H, W, C]
        diff_np = np.pad(diff_np, ((1,0),(0,0),(0,0),(0,0)), mode='constant')  # Pad to match length

        combined_np = np.concatenate([diff_np, raw_np], axis=-1)  # [T, H, W, 6]
        frames_tensor = torch.tensor(combined_np, dtype=torch.float32).permute(0, 3, 1, 2).to(self.device)  # [T, 6, H, W]

        return frames_tensor
    
    def prepare_input_for_efficientphys(self):
        frames_np = np.array(self.buffer) / 255.0  # [T, H, W, C]
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        B, C, T, H, W = frames_tensor.shape
        inputs = frames_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)  # [300, 3, 96, 96]

        return inputs
    
    def prepare_input_for_physnet(self):
        frames_np = np.array(self.buffer) / 255.0  # [T, H, W, C]
        frames_tensor = torch.tensor(frames_np, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        return frames_tensor
    
    def prep_input(self):
        if self.model_name == 'EfficientPhys':
            inputs = self.prepare_input_for_efficientphys()
            rets = -1
        elif self.model_name == 'PhysNet':
            inputs = self.prepare_input_for_physnet()
            rets = 0
        elif self.model_name == 'BigSmall':
            inputs = self.prepare_input_for_bigsmall()
            rets = 1
        elif self.model_name == 'DeepPhys':
            inputs = self.prepare_input_for_deepphys()
            rets = -1

        return inputs, rets

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Webcam read failed")
            return

        face_crop, detection = self.extract_face_crop(frame)
        # face_crop, detection = self.extract_face_regions(frame)
        if face_crop is not None:
            self.buffer.append(face_crop)
            self.mp_drawing.draw_detection(frame, detection)

        # Optional: visualize for debugging
        cv2.imshow('Face Detection', frame)
        cv2.waitKey(1)

        if len(self.buffer) >= self.window_size:
            start1 = perf_counter()
            inputs, rets = self.prep_input()
            end1 = perf_counter()
            print(f"Input Preprocess latency: {(end1 - start1)*1000:.2f} ms")
            print(f'Input Memory Usage: {self.get_memory_usage_mb()} MB')

            start2 = perf_counter()
            with torch.no_grad():
                if rets == -1:
                    rppg_signal = self.model(inputs)
                else:
                    rppg_signal = self.model(inputs)[rets]
            bvp = rppg_signal.detach().cpu().numpy().flatten()
            end2= perf_counter()
            print(f"Inference Preprocess latency: {(end2 - start2)*1000:.2f} ms")
            print(f'Inference Usage: {self.get_memory_usage_mb()}')

            start3 = perf_counter()
            print('BVP', bvp)
            bpm = _calculate_fft_hr(bvp, fs=self.fps)
            # bpm = _calculate_peak_hr(bvp, fs = self.fps)
            end3 = perf_counter()
            print(f"BPM estimate latency: {(end3 - start3)*1000:.2f} ms")
            print(f'BPM Memory Usage: {self.get_memory_usage_mb()}')
            

            start4 = perf_counter()
            self.publisher_.publish(Float32(data=bpm))
            end4 = perf_counter()
            print(f"Publisher latency: {(end4 - start4)*1000:.2f} ms")
            
            print(f"BPM Published: {bpm:.2f}")
            
            # except Exception as e:
                # self.get_logger().warn(f"Inference error: {e}")

            # Keep overlap
            self.buffer = self.buffer[int(self.window_size * 0.5):]
            # end = perf_counter()
            # print(f"Inference latency: {(end - start)*1000:.2f} ms")

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RPPGNeuralNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
