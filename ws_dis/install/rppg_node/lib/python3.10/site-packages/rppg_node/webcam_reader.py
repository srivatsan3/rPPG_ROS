import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import os
os.environ['GLOG_minloglevel'] = '2'

# node_path = '/home/mscrobotics2425laptop11/rPPG_ROS/ws_dis/src/rppg_node/'
# sys.path.insert(0, node_path) # Path to rPPG toolbox
from utils.face_roi_detection import *

mp_drawing = mp.solutions.drawing_utils

class WebcamBufferPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_buffer_publisher_node')

        self.declare_parameter('webcam_id',0)
        self.declare_parameter('frame_rate',30)
        self.declare_parameter('topic','/camera')

        self.webcam_id = self.get_parameter('webcam_id').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.publish_topic = self.get_parameter('topic').get_parameter_value().string_value
        
        # self.webcam_id = 0
        # self.frame_rate = 30
        # self.window_size_s = 5
        # self.overlap_s = 1
        # self.publish_topic = 'camera'

        print('~~~~~~~~~~~~~~~~~~~~~ PARAMS ~~~~~~~~~~~~~~~~')
        print(self.webcam_id,self.frame_rate,self.publish_topic)
        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open webcam ID {self.webcam_id}')
            return
        
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.bridge = CvBridge()
        self.frame_buffer = []

        timer_period = 1/self.frame_rate
        self.timer = self.create_timer(timer_period, self.capture_frame)
    
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed')
            return
        # print(frame)
        frame_cropped, face_landmarks = extract_face_regions(frame, roi = 'LEFT CHEEK',target_size=(96,128))
        # print(frame_cropped)

        if frame_cropped is not None:
            self.frame_buffer.append(frame_cropped)
            cv2.imshow('face',frame_cropped)
            mp_drawing.draw_landmarks(image = frame, landmark_list=face_landmarks,
            connections=None,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2))
            msg = self.bridge.cv2_to_imgmsg(frame_cropped, encoding = 'bgr8')
            self.publisher.publish(msg)  
        else:
            self.get_logger().warn("No face detected — skipping frame")

        cv2.imshow('Face ROI Viewer', frame)
        cv2.waitKey(1)

        # print('**********************LEN*******************',len(self.frame_buffer))
        # if len(self.frame_buffer) == self.window_length:
        #     # batch_msg = FrameBatch()
        #     # batch_msg.frames = [self.bridge.cv2_to_imgmsg(f, encoding='bgr8') for f in self.frame_buffer]
        #     # self.publisher.publish(batch_msg)
        #     for f in self.frame_buffer:
        #         msg = self.bridge.cv2_to_imgmsg(f, encoding = 'bgr8')
        #         self.publisher.publish(msg)

        
        # self.get_logger().info(f'Published video buffer')
        
        # self.frame_buffer = self.frame_buffer[~self.overlap_length:]
    
def main(args=None):
    rclpy.init(args=args)
    node = WebcamBufferPublisherNode()

    # try:

    rclpy.spin(node)
    # except Exception as e:
        # print(e)
        

if __name__ == '__main__':
    main()

