import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from scipy.signal import find_peaks
from time import perf_counter
import cv2
import mediapipe as mp
import numpy as np
import time
import sys
rppg_tb_path = '/home/mscrobotics2425laptop11/Dissertation/rppgtb/rPPG-Toolbox'
sys.path.insert(0, rppg_tb_path) # Path to rPPG toolbox

# Importing different non Neural Network methods from the rPPG Toolbox
from unsupervised_methods.methods.CHROME_DEHAAN import CHROME_DEHAAN
from unsupervised_methods.methods.GREEN import GREEN
from unsupervised_methods.methods.LGI import LGI
from unsupervised_methods.methods.ICA_POH import ICA_POH
from unsupervised_methods.methods.POS_WANG import POS_WANG
from unsupervised_methods.methods.OMIT import OMIT
from unsupervised_methods.methods.PBV import PBV
from unsupervised_methods.methods.PBV import PBV2

# Importing Post processing tools from rPPG Toolbox
from evaluation.post_process import _calculate_peak_hr, _calculate_fft_hr

mp_face_detection = mp.solutions.face_detection #Mediapipe face detection
mp_drawing = mp.solutions.drawing_utils

def extract_face_roi(frame, box_size=128, face_detection = None):
    '''
    Function to extract the region of interest (Face)

    Parameters:
    frame : An array representing a single frame of the video
    box_size: The size of region of interest required
    face_detection : Face Dectection method

    Returns:
    face_crop : Cropped image with detected face in it
    detection : Key points detection and boundix box information
    '''
    if face_detection is None:
        return None
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None,None  # No face detected

    # Use first detected face
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


class RPPGToolboxNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_node')
        self.publisher_ = self.create_publisher(msg_type = Float32, topic = 'heart_rate_bpm', qos_profile = 10)
        self.face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)

        self.buffer = []
        self.fps = 30
        self.total_window_secs = 10
        self.overlap_secs = 9

        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        

        self.window_size = int(self.total_window_secs * self.fps) 
        self.extra_frames = int((self.total_window_secs - self.overlap_secs) * self.fps)

    def estimate_bpm_from_bvp(self,bvp, fps):
        peaks, _ = find_peaks(bvp, distance=fps * 0.5)  
        duration_sec = len(bvp) / fps
        bpm = (len(peaks) / duration_sec) * 60
        return bpm


    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame from webcam")
            return
        frame_cropped, detection = extract_face_roi(frame, face_detection=self.face_detector)

        if frame_cropped is not None:
            self.buffer.append(frame_cropped)
            mp_drawing.draw_detection(frame, detection)
        else:
            self.get_logger().warn("No face detected — skipping frame")

        cv2.imshow('Face ROI Viewer', frame)
        cv2.waitKey(1)

        if frame_cropped is not None:
            self.buffer.append(frame_cropped)

        if len(self.buffer) >= self.window_size:
            start = perf_counter()

            try:
                bvp = CHROME_DEHAAN(self.buffer, FS=self.fps)
                # bvp = POS_WANG(self.buffer, fs=self.fps)
                # bvp = ICA_POH(self.buffer, FS = self.fps)

                # bvp = GREEN(self.buffer)
                # bvp = PBV(self.buffer)
                # bvp  = PBV2(self.buffer)
                # bvp = LGI(self.buffer)
                # bvp = OMIT(self.buffer)
                

                # Post-process BVP to get BPM
                bpm = self.estimate_bpm_from_bvp(bvp, fps=self.fps)
                # bpm = _calculate_fft_hr(bvp, fs = self.fps)
                # bpm = _calculate_peak_hr(bvp, fs = self.fps)


                # Publish BPM
                self.publisher_.publish(Float32(data=bpm))
                print(f"Published BPM: {bpm:.2f}")
            except Exception as e:
                self.get_logger().warn(f"CHROM processing error: {e}")
            end = perf_counter()
            print(f"Inference latency: {(end - start)*1000:.2f} ms")

            self.buffer = self.buffer[self.extra_frames:]

def main(args=None):
    rclpy.init(args=args)
    node = RPPGToolboxNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
