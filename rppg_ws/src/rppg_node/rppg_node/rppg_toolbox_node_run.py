import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
import mediapipe as mp
from utils.face_roi_detection import *
import cv2


from utils.rppg_utils import *
NN_ALGOS = ['physnet','efficientphys','deepphys','bigsmall'] # List of algorithms that use neural networks
mp_drawing = mp.solutions.drawing_utils

class RPPGToolboxNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_node')

        self.declare_parameter('frame_rate',30)             # Frame rate for processing
        self.declare_parameter('camera_topic','/camera')    # Topic to subscribe to for camera frames
        self.declare_parameter('window_secs', 8)            # Duration of the window for processing in seconds
        self.declare_parameter('overlap_secs', 6)           # Overlap duration between consecutive windows in seconds
        self.declare_parameter('roi_area', 'all')
        self.declare_parameter('img_width', 128)
        self.declare_parameter('img_height', 128)
        

        self.declare_parameter('topic','/heart_rate_bpm')   # Topic to publish the heart rate in BPM
        self.declare_parameter('algo','deepphys')           # Algorithm to use for rPPG processing
        self.declare_parameter('estimate','fft')            # Method to estimate BPM ('fft' or 'peak')
        self.declare_parameter('viz',True)
 
        
        self.fps = self.get_parameter('frame_rate').get_parameter_value().integer_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.window_size_s = self.get_parameter('window_secs').get_parameter_value().integer_value
        self.overlap_s = self.get_parameter('overlap_secs').get_parameter_value().integer_value
        self.roi_area = self.get_parameter('roi_area').get_parameter_value().string_value
        self.img_width = self.get_parameter('img_width').get_parameter_value().integer_value
        self.img_height = self.get_parameter('img_height').get_parameter_value().integer_value
        self.viz = self.get_parameter('viz').get_parameter_value().bool_value
        

        self.algo = self.get_parameter('algo').get_parameter_value().string_value
        self.bpm_estimate = self.get_parameter('estimate').get_parameter_value().string_value
        self.publish_topic = self.get_parameter('topic').get_parameter_value().string_value

        self.window_length = int(self.fps*self.window_size_s)   # Length of the window in frames
        self.overlap_length = int(self.fps*self.overlap_s)      # Length of the overlap in frames

        self.subscription = self.create_subscription(Image, self.camera_topic, self.frame_callback, 10) # Subscription to camera topic
        self.publisher_ = self.create_publisher(msg_type = Float32, topic = 'heart_rate_bpm', qos_profile = 10) # Publisher for heart rate
        self.bridge = CvBridge()
        self.roi_area = self.roi_area.replace('_',' ').upper()
        self.frame_buffer = deque(maxlen=self.window_length)    # Buffer to hold frames for processing

        if self.algo in NN_ALGOS:
            self.model, checkpoint_path = load_model(algo = self.algo, frames = self.window_length) # Load the Neural Netowrks model
            
            state_dict = torch.load(checkpoint_path, map_location='cpu') 
            cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()} 
            self.model.load_state_dict(cleaned_state, strict = False)
            self.model.eval() # Set the model to evaluation mode

    def frame_callback(self,msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # Convert ROS Image message to OpenCV format
        if self.roi_area != 'ALL':
            frame_cropped, face_landmarks = extract_face_regions(frame, roi = self.roi_area, target_size=(self.img_width,self.img_height))
        else:
            frame_cropped, detection = extract_face(frame, box_size=(self.img_width, self.img_height))


        if frame_cropped is not None:
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
            self.frame_buffer.append(frame_cropped)  
        else:
            self.get_logger().warn("No face detected — skipping frame")

        if self.viz:
            cv2.imshow('Face ROI Viewer', frame)
            cv2.waitKey(1)

        if len(self.frame_buffer) == self.window_length:   # Check if the buffer is full
            if self.algo not in NN_ALGOS:                  # If the algorithm is not a neural network
                bpm = run_rppg(buffer = self.frame_buffer, 
                               fps = self.fps ,
                               algo = self.algo, 
                               bpm_estimate=self.bpm_estimate)
            else:
                bpm = run_rppg_nn(buffer = self.frame_buffer, 
                                  fps = self.fps, 
                                  algo = self.algo, 
                                  bpm_estimate = self.bpm_estimate, 
                                  model = self.model)
            
            self.publisher_.publish(Float32(data=float(bpm)))   # Publish the estimated BPM
            print(f"Published BPM ({self.algo}): {bpm:.2f}")    

            for _ in range(self.window_length - self.overlap_length): # Remove frames from the buffer to maintain overlap
                self.frame_buffer.popleft()


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
