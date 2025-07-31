# Remote PolyPythesmoGraphy (rPPG) tool blended with ROS2 setup

## ROS2 Nodes:
The Robot Operating System 2 **Humble** version is used throughout.
Each of the Nodes perform specific rPPG based algorithms and publish them to a ROS2 Topic
The nodes:

- **rppg_toolbox_node.py** : A Python Node that runs classical algorithms mentioned in the rppg tool box over a live webcam based feed for real-time publishing.
- **rppg_toolbox_nn_node.py** : A Python Node that runs neural network algorithms mentioned in the rppg tool box over a live webcam based feed for real-time publishing.
- **rppg_toolbox_video_node.py** : A Python Node that runs classical algorithms mentioned in the rppg tool box over a custom video (supports .avi and .mat files).
- **wrapper.py**: A wrapper function to run one of the above mentioned nodes via the *ros2 run rppg_node rppg_node* command
