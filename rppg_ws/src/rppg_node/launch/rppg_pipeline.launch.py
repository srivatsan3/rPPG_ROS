from launch import LaunchDescription
from launch_ros.actions import Node
import os
import yaml

def generate_launch_description():
    ''' Generates launch description for the rPPG pipeline with webcam reader and video node. '''

    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir,'config')
    rppg_params = os.path.join(config_dir, 'rppg_run_params.yaml')  # Path to rPPG parameters

    with open(rppg_params,'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
    python_path = data['rppg_toolbox_node']['ros__parameters']['python_path']

    # Launch description for the webcam reader and rPPG video node
    launch_desc = LaunchDescription([
        Node(
            package = 'rppg_node',
            executable='webcam_reader',
            name = 'webcam_buffer_publisher_node',
            parameters = [rppg_params],
            output = 'screen',
            prefix=python_path
        ),
        Node(
            package = 'rppg_node',
            executable='rppg_toolbox_node_run',
            name = 'rppg_toolbox_node',
            parameters = [rppg_params],
            output = 'screen',
            prefix=python_path
        )
    ])

    return launch_desc

