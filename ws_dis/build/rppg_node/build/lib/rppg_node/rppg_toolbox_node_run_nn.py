import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from time import perf_counter
import cv2
import mediapipe as mp
import numpy as np
import torch
import sys
import os
import psutil
import contextlib
from cv_bridge import CvBridge

utils_path = '/home/mscrobotics2425laptop11/rPPG_ROS/ws_dis/src/rppg_node'
sys.path.insert(0, utils_path) # Path to rPPG toolbox
from utils.rppg_utils import *


class RPPGNeuralNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_node')

        self.declare_parameter('frame_rate',30)
        self.declare_parameter('camera_topic','/camera')
        self.declare_parameter('window_secs', 8)
        self.declare_parameter('overlap_secs', 2)

        self.declare_parameter('topic','/heart_rate_bpm')
        self.declare_parameter('algo','physnet')
        self.declare_parameter('estimate','fft')
 

        self.fps = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.window_size_s = self.get_parameter('window_secs').get_parameter_value().integer_value
        self.overlap_s = self.get_parameter('overlap_secs').get_parameter_value().integer_value

        self.algo = self.get_parameter('algo').get_parameter_value().string_value
        self.bpm_estimate = self.get_parameter('estimate').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('topic').get_parameter_value().string_value

        self.window_length = int(self.fps*self.window_size_s)
        self.overlap_length = int(self.fps*self.overlap_s)
        self.img_size = 128

        self.subscription = self.create_subscription(Image, self.camera_topic, self.frame_callback, 10)
        self.publisher_ = self.create_publisher(msg_type = Float32, topic = 'heart_rate_bpm', qos_profile = 10)
        # self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.bridge = CvBridge()
        self.frame_buffer = []

        print('~~~~~~~~~~~~~~~~~~~~~ PARAMS ~~~~~~~~~~~~~~~~')
        print(self.fps,self.camera_topic,self.algo,self.bpm_estimate)

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, checkpoint_path = load_model(algo = self.algo, frames = self.window_length, img_size = self.img_size)
        
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(cleaned_state)
        self.model.eval()


    def get_memory_usage_mb(self):
        process = psutil.Process(os.getpid())
        mem_bytes = process.memory_info().rss  # Resident Set Size in bytes
        return mem_bytes / (1024 ** 2)  # Convert to MB


    def frame_callback(self,msg):
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding = 'bgr8')
        self.frame_buffer.append(frame)


        if len(self.frame_buffer) == self.window_length:
            bpm = run_rppg_nn(buffer = self.frame_buffer, 
                              fps = self.fps, 
                              window_size = self.window_length, 
                              roi_size = self.img_size, 
                              model = self.model,
                              algo = self.algo, 
                              bpm_estimate = self.bpm_estimate,
                              device = self.device)

            self.publisher_.publish(Float32(data=bpm))

            print(f"BPM Published: {bpm:.2f}")
            
            
            self.frame_buffer = self.frame_buffer[~self.overlap_length:]
            # print(f"Inference latency: {(end - start)*1000:.2f} ms")



def main(args=None):
    rclpy.init(args=args)
    node = RPPGNeuralNode()
    rclpy.spin(node)
    # node.cap.release()
    # node.destroy_node()
    # cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
