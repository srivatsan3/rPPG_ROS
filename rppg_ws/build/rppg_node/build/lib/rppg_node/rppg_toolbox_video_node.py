import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from time import perf_counter
import cv2
import mediapipe as mp
import numpy as np
import os
from utils.rppg_utils import *
from utils.face_roi_detection import *
from utils.utils import read_video_stream
from collections import deque

mp_drawing = mp.solutions.drawing_utils
NN_ALGOS = ['physnet','efficientphys','deepphys','bigsmall']

class RPPGVideoNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_video_node')
        
        self.declare_parameter('frame_rate',30)
        self.declare_parameter('window_secs', 8)
        self.declare_parameter('overlap_secs', 6)
        self.declare_parameter('video_path', '/home/mscrobotics2425laptop11/Dissertation/UBFC/RawData/subject1/vid.avi')
        self.declare_parameter('dict_key', '')
        self.declare_parameter('img_width', 128)
        self.declare_parameter('img_height', 128)
        self.declare_parameter('roi_area','all')
        self.declare_parameter('viz',True)

        self.declare_parameter('topic','/heart_rate_bpm')
        self.declare_parameter('algo','deepphys')
        self.declare_parameter('estimate','fft')
 
        self.fps = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.window_size_s = self.get_parameter('window_secs').get_parameter_value().integer_value
        self.overlap_s = self.get_parameter('overlap_secs').get_parameter_value().integer_value
        self.video_path = self.get_parameter('video_path').get_parameter_value().string_value
        self.dict_key = self.get_parameter('dict_key').get_parameter_value().string_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('img_height').get_parameter_value().integer_value
        self.roi_area = self.get_parameter('roi_area').get_parameter_value().string_value
        self.viz = self.get_parameter('viz').get_parameter_value().bool_value

        self.algo = self.get_parameter('algo').get_parameter_value().string_value
        self.bpm_estimate = self.get_parameter('estimate').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('topic').get_parameter_value().string_value

        self.window_length = int(self.fps*self.window_size_s)
        self.overlap_length = int(self.fps*self.overlap_s)
        self.roi_area = self.roi_area.replace('_',' ').upper()
        self.frame_buffer = deque(maxlen=self.window_length)
        self.frame_index = 0
        
        if self.algo in NN_ALGOS:
            self.model, checkpoint_path = load_model(algo = self.algo, frames = self.window_length)
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state)
            self.model.eval()

        self.publisher_ = self.create_publisher(Float32, self.publish_topic, 10)
        self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)

    def timer_callback(self):

        for frame in read_video_stream(self.video_path, self.dict_key):
            self.frame_index += 1
            if self.roi_area != 'ALL':
                frame_cropped, face_landmarks = extract_face_regions(frame, roi = self.roi_area,target_size=(self.img_width, self.img_height))
            else:
                frame_cropped, detection = extract_face(frame = frame, box_size= (self.img_width, self.img_height))
            if frame_cropped is not None:
                self.frame_buffer.append(frame_cropped)
                if self.viz:
                    cv2.imshow('Region of Interest',frame_cropped)

                if self.roi_area != 'ALL':
                    if self.viz:
                        mp_drawing.draw_landmarks(image = frame, landmark_list=face_landmarks,
                        connections=None,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2))
                    del face_landmarks
                else:
                    if self.viz:
                        mp_drawing.draw_detection(frame, detection)
                    del detection
            else:
                self.get_logger().warn("No face detected : skipping frame")

            if self.viz:
                cv2.imshow('Face ROI Viewer', frame)
                cv2.waitKey(1)

            if len(self.frame_buffer) == self.window_length:
                try:
                    if self.algo not in NN_ALGOS:
                        bpm = run_rppg(buffer = self.frame_buffer, fps = self.fps ,algo = self.algo, bpm_estimate=self.bpm_estimate)
                    else:
                        bpm = run_rppg_nn(buffer = self.frame_buffer, fps = self.fps, algo = self.algo, bpm_estimate = self.bpm_estimate, model = self.model)
                
                    self.publisher_.publish(Float32(data=bpm))
                    print(f"Published BPM ({self.algo}): {bpm:.2f}")
                except Exception as e:
                    self.get_logger().warn(f"Processing error: {e}")

                for _ in range(self.window_length - self.overlap_length):
                    self.frame_buffer.popleft()
                    
def main(args=None):
    rclpy.init(args=args)
    node = RPPGVideoNode()
    rclpy.spin(node)
    node.cap.release()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
