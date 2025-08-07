import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
from processing.face_roi_detection import *

mp_drawing = mp.solutions.drawing_utils

class WebcamBufferPublisherNode(Node):
    def __init__(self):
        super().__init__('webcam_buffer_publisher_node')

        self.webcam_id = self.get_parameter_or('webcam_id',0).get_parameter_value().integer_value
        self.frame_rate = self.get_parameter_or('frame_rate',0).get_parameter_value().integer_value
        self.window_size_s = self.get_parameter_or('window_size',0).get_parameter_value().double_value
        self.window_length = int(self.frame_rate*self.window_size_s)
        self.overlap_s = self.get_parameter_or('overlap_size',0).get_parameter_value().double_value
        self.overlap_length = int(self.frame_rate*self.overlap_s)
        self.publish_topic = self.get_parameter_or('topic','camera').get_parameter_value().string_value

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

        frame_cropped, face_landmarks = extract_face_regions(frame)


        if frame_cropped is not None:
            self.frame_buffer.append(frame_cropped)
            cv2.imshow('face',frame_cropped)
            mp_drawing.draw_landmarks(image = frame, landmark_list=face_landmarks,
            connections=None,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2))
        else:
            self.get_logger().warn("No face detected — skipping frame")

        cv2.imshow('Face ROI Viewer', frame)
        cv2.waitKey(1)


        if len(self.frame_buffer) == self.window_length:

            for f in self.frame_buffer:
                msg = self.bridge.cv2_to_imgmsg(f, encoding = 'bgr8')
                self.publisher.publish(msg)
            
            self.get_logger().info(f'Published video buffer')
            
            self.frame_buffer = self.frame_buffer[~self.overlap_length:]
    
def main(args=None):
    rclpy.init(args=args)
    node = WebcamBufferPublisherNode()

    try:
        rclpy.spin(node)
    except:
        pass

if __name__ == '__main__':
    main()

