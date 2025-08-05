import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from scipy.signal import find_peaks
from time import perf_counter
import cv2
import mat73
import mediapipe as mp
import numpy as np
import sys
import scipy
rppg_tb_path = '/home/mscrobotics2425laptop11/Dissertation/rppgtb/rPPG-Toolbox'
sys.path.insert(0, rppg_tb_path) 

# Classical methods
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from evaluation.post_process import _calculate_peak_hr, _calculate_fft_hr

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_face_roi(frame, box_size=128, face_detection=None):
    if face_detection is None:
        return None, None
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.detections:
        return None, None
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = frame.shape
    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)
    face_crop = frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (box_size, box_size))
    return face_crop, detection

class RPPGVideoNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_video_node')
        self.publisher_ = self.create_publisher(Float32, 'heart_rate_bpm', 10)
        self.video_path = '/home/mscrobotics2425laptop11/Dissertation/scamps_sample/P000001.mat'
        
        if self.video_path.split('.')[-1] == 'mat':
            data = mat73.loadmat(self.video_path)
            print(data.keys())
            self.video_frames = data['RawFrames']  # Replace with actual key

        # for frame in video_frames:
        #     face_crop, _ = extract_face_roi(frame, face_detection=self.face_detector)
        #     if face_crop is not None:
        #         self.buffer.append(face_crop)
        # self.cap = cv2.VideoCapture(self.video_path)

        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.buffer = []
        self.frame_index = 0
        self.total_frames  = self.video_frames.shape[0]
        # self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.fps = 30
        self.total_window_secs = 10
        self.window_size = int(self.total_window_secs * self.fps)
        self.overlap_secs = 9
        self.extra_frames = int((self.total_window_secs - self.overlap_secs) * self.fps)

        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

    def estimate_bpm_from_bvp(self, bvp, fps):
        peaks, _ = find_peaks(bvp, distance=fps * 0.5)
        duration_sec = len(bvp) / fps
        bpm = (len(peaks) / duration_sec) * 60
        return bpm

    def timer_callback(self):
        # ret, frame = self.cap.read()
        # for frame in self.video_frames:
        #     face_crop, _ = extract_face_roi(frame, face_detection=self.face_detector)
        #     if face_crop is not None:
        #         self.buffer.append(face_crop)

        # if not ret:
        #     self.get_logger().info("Video stream finished.")
        #     self.destroy_node()
        #     return
        if self.frame_index >= self.total_frames:
            self.get_logger().info(".mat file stream finished.")
            self.destroy_node()
            return

        frame = self.video_frames[self.frame_index]
        # frame = frame.astype(np.uint8)  # If values are in 0–255 range
        frame = (frame * 255).clip(0, 255).astype(np.uint8)


        self.frame_index += 1
        face_crop, detection = extract_face_roi(frame, face_detection=self.face_detector)
        if face_crop is not None:
            self.buffer.append(face_crop)

        if len(self.buffer) >= self.window_size:
            try:
                start = perf_counter()
                bvp = CHROME_DEHAAN(self.buffer, FS=self.fps)
                bpm = self.estimate_bpm_from_bvp(bvp, self.fps)
                self.publisher_.publish(Float32(data=bpm))
                print(f"Published BPM (CHROM): {bpm:.2f}")
                print(f"Inference latency: {(perf_counter() - start)*1000:.2f} ms")
            except Exception as e:
                self.get_logger().warn(f"CHROM processing error: {e}")

            self.buffer = self.buffer[self.extra_frames:]

def main(args=None):
    rclpy.init(args=args)
    node = RPPGVideoNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
