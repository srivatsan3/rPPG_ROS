import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import os
from time import perf_counter
from datetime import datetime

from utils.face_roi_detection import *

mp_drawing = mp.solutions.drawing_utils

class WebcamBufferPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_buffer_publisher_node')

        self.declare_parameter('webcam_id',0)
        self.declare_parameter('frame_rate',30)
        self.declare_parameter('camera_topic','/camera')

        self.webcam_id = self.get_parameter('webcam_id').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.publish_topic = self.get_parameter('camera_topic').get_parameter_value().string_value


        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open webcam ID {self.webcam_id}')
            return
        
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.bridge = CvBridge()

        timer_period = 1/self.frame_rate
        self.timer = self.create_timer(timer_period, self.capture_frame)
    
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed')
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
        self.publisher.publish(msg)  

def main(args=None):
    rclpy.init(args=args)
    node = WebcamBufferPublisherNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

