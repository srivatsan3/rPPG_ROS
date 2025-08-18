from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os

def generate_launch_description():

    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_dir = os.path.join(base_dir,'config')

    webcam_params = os.path.join(config_dir, 'webcam_params.yaml')
    rppg_params = os.path.join(config_dir, 'rppg_run_params.yaml')

    python_path = '/home/mscrobotics2425laptop11/rppg_env/bin/python'
    webcam_node = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'{python_path} -m rppg_node.webcam_reader --ros-args -p config:={webcam_params} 2>/dev/null'
        ],
        output='screen'
    )

    rppg_toolbox_node = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'{python_path} -m rppg_node.rppg_toolbox_node_run --ros-args -p config:={rppg_params} 2>/dev/null'
        ],
        output='screen'
    )

    system_monitor_node = ExecuteProcess(
        cmd=[
            'bash', '-c',
            f'{python_path} -m rppg_node.system_monitor'
        ],
        output='screen'
    )
    return LaunchDescription([
        webcam_node,
        rppg_toolbox_node,
        # Node(
        #     package='rppg_node',
        #     executable='system_monitor',
        #     name='system_monitor',
        #     output='screen',
        #     prefix='/home/mscrobotics2425laptop11/rppg_env/bin/python'
        # )
    ])
    launch_desc = LaunchDescription([
        Node(
            package = 'rppg_node',
            executable='webcam_reader',
            name = 'webcam_buffer_publisher_node',
            parameters = [webcam_params],
            output = 'screen',
            prefix='/home/mscrobotics2425laptop11/rppg_env/bin/python'
        ),
        Node(
            package = 'rppg_node',
            executable='rppg_toolbox_node_run',
            name = 'rppg_toolbox_node',
            parameters = [rppg_params],
            output = 'screen',
            prefix='/home/mscrobotics2425laptop11/rppg_env/bin/python'
        ),
        # Node(
        #     package='rppg_node',
        #     executable='system_monitor',
        #     name='system_monitor',
        #     output='screen',
        #     prefix='/home/mscrobotics2425laptop11/rppg_env/bin/python'
        # )


    ])

    return launch_desc

