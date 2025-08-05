import sys
print("Python interpreter:", sys.executable)

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import tempfile
import os
import time
import sys
sys.path.insert(0, '/home/mscrobotics2425laptop11/Dissertation/pyVHR-pyVHR_CPU')  # Adjust if needed

from pyVHR.analysis.pipeline import Pipeline

class RPPGNode(Node):
    def __init__(self):
        super().__init__('rppg_node')
        print('**************************init 1passed**************** \n\n')
        self.publisher_ = self.create_publisher(Float32, 'heart_rate_bpm', 10)
        self.cap = cv2.VideoCapture(0)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Camera FPS: {fps}")

        self.buffer = []
        # self.pipe = Pipeline(method='cpu_CHROM', roi_approach='patches', wsize=6)
        self.pipe  = Pipeline()
        self.timer = self.create_timer(1/30, self.timer_callback)
    
    def process_buffer_with_pyVHR(self):
        # Create a temporary video file
        temp_video = tempfile.NamedTemporaryFile(suffix='.avi', delete=False)
        temp_path = temp_video.name
        print(temp_path)
        temp_video.close()

        # Write frames to the video
        height, width, _ = self.buffer[0].shape
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        for frame in self.buffer:
            out.write(frame)
        out.release()

        # Run pyVHR on the temporary video
        t, bpm_array,bpm_mad, bvps = self.pipe.run_on_video(temp_path,
                # '/home/mscrobotics2425laptop11/Dissertation/UBFC/subject1/vid.avi',
                method='cpu_'+'CHROM',
                roi_method='convexhull',
                roi_approach='patches',
                pre_filt=True,
                post_filt=True,
                verb = False
            )

        # Clean up
        # os.remove(temp_path)

        return bpm_array
        

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to read frame")
            return
        self.buffer.append(frame)
        if len(self.buffer) % 30 == 0:
            print(len(self.buffer))
        if len(self.buffer) >= 10*30:
            # print(self.buffer)
            # bpm = self.pipe.run(self.buffer)
            # t, bpm_array,bpm_mad, bvps = self.pipe.run_on_video(self.buffer,
            #     # '/home/mscrobotics2425laptop11/Dissertation/UBFC/subject1/vid.avi',
            #     method='cpu_'+'CHROM',
            #     roi_method='convexhull',
            #     roi_approach='hol',
            #     pre_filt=True,
            #     post_filt=True
            # )
            bpm_array = self.process_buffer_with_pyVHR()
            print(bpm_array)
            # print(bpm)
            # print(type(bpm))
            for bpm in bpm_array:
                self.publisher_.publish(Float32(data=bpm))
                self.get_logger().info(f"Published BPM: {bpm:.2f}")
                time.sleep(1.0 / 30)
            # result = self.pipe.run_on_video('/home/mscrobotics2425laptop11/Dissertation/UBFC/subject1/vid.avi', roi_approach="hol", roi_method="faceparsing", cuda=False, method = 'cpu_CHROM')
            # print(result)
            # print(type(result))
            # bpm = result
            # self.publisher_.publish(Float32(data=bpm))
            # self.get_logger().info(f'Published BPM: {bpm:.2f}')
            self.buffer = []

def main(args=None):
    print('**************************main passed**************** \n\n')
    rclpy.init(args=args)
    node = RPPGNode()
    rclpy.spin(node)
    node.cap.release()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
