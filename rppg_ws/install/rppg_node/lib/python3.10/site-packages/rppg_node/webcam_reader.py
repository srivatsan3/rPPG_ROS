import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import mediapipe as mp
import gc
import os
from time import perf_counter
from datetime import datetime
import psutil
os.environ['GLOG_minloglevel'] = '2'
import objgraph
import tracemalloc

from utils.face_roi_detection import *

mp_drawing = mp.solutions.drawing_utils
tracemalloc.start()

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
        
        # self.webcam_id = 0
        # self.frame_rate = 30
        # self.window_size_s = 5
        # self.overlap_s = 1
        # self.publish_topic = 'camera'

        print('~~~~~~~~~~~~~~~~~~~~~ PARAMS ~~~~~~~~~~~~~~~~')
        print(self.webcam_id,self.frame_rate,self.publish_topic, self.img_height, self.roi_area)
        self.cap = cv2.VideoCapture(self.webcam_id)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open webcam ID {self.webcam_id}')
            return
        
        self.publisher = self.create_publisher(Image, self.publish_topic, 10)
        self.bridge = CvBridge()
        self.roi_area = self.roi_area.replace('_',' ').upper()

        timer_period = 1/self.frame_rate
        self.timer = self.create_timer(timer_period, self.capture_frame)

        self.log_to_csv = False
        self.csv_path = 'inference_metrics.csv'
        if self.log_to_csv and not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write(','.join(['img_size','roi',
                    'timestamp','latency','ram'
                ]) + '\n')
    
    def capture_frame(self):
        start = perf_counter()
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Frame capture failed')
            return
        # print(frame)
        if self.roi_area != 'ALL':
            frame_cropped, face_landmarks = extract_face_regions(frame, roi = self.roi_area, target_size=(self.img_width,self.img_height))
        else:
            frame_cropped, detection = extract_face(frame, box_size=(self.img_width, self.img_height))
        # print(frame_cropped)

        if frame_cropped is not None:
            cv2.imshow('face',frame_cropped)
            if self.roi_area != 'ALL':
                mp_drawing.draw_landmarks(image = frame, landmark_list=face_landmarks,
                connections=None,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=2))
                face_landmarks = None
                del face_landmarks
            else:
                mp_drawing.draw_detection(frame, detection)
                detection = None
                del detection
            msg = self.bridge.cv2_to_imgmsg(frame_cropped, encoding = 'bgr8')
            self.publisher.publish(msg)  
        else:
            self.get_logger().warn("No face detected — skipping frame")

        cv2.imshow('Face ROI Viewer', frame)
        cv2.waitKey(1)

        end = perf_counter()

        
        # print(f"Frame Inference latency: {(end - start)*1000:.2f} ms")
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        # print(f' Memory Consumption',mem_info.rss / (1024 ** 2))

        timestamp = datetime.utcnow().isoformat()

        metrics = {
            'img_size' : self.img_height,
            'roi' : self.roi_area,
            'timestamp': timestamp,
            'latency' : (end - start)*1000,
            'ram': mem_info.rss / (1024 ** 2)
        }
        if self.log_to_csv:
            with open(self.csv_path, 'a') as f:
                f.write(','.join(str(metrics[k]) for k in metrics) + '\n')

        # del process, mem_info, timestamp,frame_cropped, frame, ret,start, end,msg
        # gc.collect()
        # # import objgraph
        # objgraph.show_growth(limit=10)
        # # objgraph.show_backrefs(objgraph.by_type('FaceDetection')[0], max_depth=3)

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 5 memory-consuming lines ]")
        # for stat in top_stats[:5]:
        #     print(stat)

    
def main(args=None):
    rclpy.init(args=args)
    node = WebcamBufferPublisherNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()
    # except Exception as e:
        # print(e)
        

if __name__ == '__main__':
    main()

