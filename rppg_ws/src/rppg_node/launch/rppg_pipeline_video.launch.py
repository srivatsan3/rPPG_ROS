from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir,'config')

    rppg_params = os.path.join(config_dir, 'rppg_run_params.yaml')
    python_path = '/home/mscrobotics2425laptop11/rppg_env/bin/python'

    launch_desc = LaunchDescription([
        Node(
            package = 'rppg_node',
            executable='rppg_toolbox_video_node',
            name = 'rppg_toolbox_node',
            parameters = [rppg_params],
            output = 'screen',
            prefix=python_path
        )
    ])

    return launch_desc

