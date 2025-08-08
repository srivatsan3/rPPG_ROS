import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from scipy.signal import find_peaks
from mediapipe.framework.formats import landmark_pb2
from sensor_msgs.msg import Image
from time import perf_counter
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import warnings
warnings.filterwarnings("ignore")
import os
import contextlib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

utils_path = '/home/mscrobotics2425laptop11/rPPG_ROS/ws_dis/src/rppg_node'
sys.path.insert(0, utils_path) # Path to rPPG toolbox
from utils.rppg_utils import *


class RPPGToolboxNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_node')

        self.declare_parameter('frame_rate',30)
        self.declare_parameter('camera_topic','/camera')
        self.declare_parameter('window_secs', 8)
        self.declare_parameter('overlap_secs', 2)

        self.declare_parameter('topic','/heart_rate_bpm')
        self.declare_parameter('algo','pos')
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


        self.subscription = self.create_subscription(Image, self.camera_topic, self.frame_callback, 10)
        self.publisher_ = self.create_publisher(msg_type = Float32, topic = 'heart_rate_bpm', qos_profile = 10)
        # self.timer = self.create_timer(1.0 / self.fps, self.timer_callback)
        self.bridge = CvBridge()
        self.frame_buffer = []

        print('~~~~~~~~~~~~~~~~~~~~~ PARAMS ~~~~~~~~~~~~~~~~')
        print(self.fps,self.camera_topic,self.algo,self.bpm_estimate)
        

        # self.algo = 'pos'
        # self.bpm_estimate = 'fft'

    def frame_callback(self,msg):
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_buffer.append(frame)

        print('******************************************',len(self.frame_buffer),'*************************************')
        if len(self.frame_buffer) == self.window_length:
            print('*******************************',len(self.frame_buffer),self.frame_buffer[0].shape,'*********************************************')
            start = perf_counter()

            bpm = run_rppg(buffer = self.frame_buffer, fps = self.fps ,algo = self.algo, bpm_estimate=self.bpm_estimate)

            print('****************************BPM*********************************',bpm)
            self.publisher_.publish(Float32(data=float(bpm)))
                
            end = perf_counter()
            print(f"Inference latency: {(end - start)*1000:.2f} ms")
            self.frame_buffer = self.frame_buffer[~self.overlap_length:]

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
