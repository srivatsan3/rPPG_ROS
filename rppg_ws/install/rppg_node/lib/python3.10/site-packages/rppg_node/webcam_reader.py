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
        self.declare_parameter('topic','/camera')
        self.declare_parameter('img_width', 128)
        self.declare_parameter('img_height', 128)
        self.declare_parameter('roi_area','all')

        self.webcam_id = self.get_parameter('webcam_id').get_parameter_value().integer_value
        self.frame_rate = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.publish_topic = self.get_parameter('topic').get_parameter_value().string_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('img_height').get_parameter_value().integer_value
        self.roi_area = self.get_parameter('roi_area').get_parameter_value().string_value

        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open webcam ID {self.webcam_id}')
            return
        
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.bridge = CvBridge()
        self.roi_area = self.roi_area.replace('_',' ').upper()

        timer_period = 1/self.frame_rate
        self.timer = self.create_timer(timer_period, self.capture_frame)
    
    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed')
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding = 'bgr8')
        self.publisher.publish(msg)  

        
        # if self.roi_area != 'ALL':
        #     frame_cropped, face_landmarks = extract_face_regions(frame, roi = self.roi_area, target_size=(self.img_width,self.img_height))
        # else:
        #     frame_cropped, detection = extract_face(frame, box_size=(self.img_width, self.img_height))


        # if frame_cropped is not None:
        #     cv2.imshow('face',frame_cropped)
        #     if self.roi_area != 'ALL':
        #         mp_drawing.draw_landmarks(image = frame, landmark_list=face_landmarks,
        #         connections=None,
        #         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2))
        #         face_landmarks = None
        #         del face_landmarks
        #     else:
        #         mp_drawing.draw_detection(frame, detection)
        #         detection = None
        #         del detection
        #     msg = self.bridge.cv2_to_imgmsg(frame_cropped, encoding = 'bgr8')
        #     self.publisher.publish(msg)  
        # else:
        #     self.get_logger().warn("No face detected — skipping frame")

        # cv2.imshow('Face ROI Viewer', frame)
        # cv2.waitKey(1)

        # end = perf_counter()
        # if end - start < 1/self.frame_rate:
        #     sleep_time = (1/self.frame_rate) - (end - start)
        #     cv2.waitKey(int(sleep_time * 1000))


    
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

