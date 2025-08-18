import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
import cv2

from utils.rppg_utils import *
NN_ALGOS = ['physnet','efficientphys','deepphys','bigsmall']

class RPPGToolboxNode(Node):
    def __init__(self):
        super().__init__('rppg_toolbox_node')

        self.declare_parameter('frame_rate',30)
        self.declare_parameter('camera_topic','/camera')
        self.declare_parameter('window_secs', 8)
        self.declare_parameter('overlap_secs', 6)
        

        self.declare_parameter('topic','/heart_rate_bpm')
        self.declare_parameter('algo','deepphys')
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
        self.bridge = CvBridge()
        self.frame_buffer = deque(maxlen=self.window_length)

        if self.algo in NN_ALGOS:
            self.model, checkpoint_path = load_model(algo = self.algo, frames = self.window_length)
            
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            cleaned_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state, strict = False)
            self.model.eval()

    def frame_callback(self,msg):
        
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_buffer.append(frame)

        if len(self.frame_buffer) == self.window_length:
            if self.algo not in NN_ALGOS:
                bpm = run_rppg(buffer = self.frame_buffer, fps = self.fps ,algo = self.algo, bpm_estimate=self.bpm_estimate)
            else:
                bpm = run_rppg_nn(buffer = self.frame_buffer, fps = self.fps, algo = self.algo, bpm_estimate = self.bpm_estimate, model = self.model)
            
            self.publisher_.publish(Float32(data=float(bpm)))
            print(f"Published BPM ({self.algo}): {bpm:.2f}")

            for _ in range(self.window_length - self.overlap_length):
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
