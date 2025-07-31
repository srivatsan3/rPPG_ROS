import subprocess
import os
import sys

def main():
    # Path to your virtualenv Python
    # venv_python = "/home/mscrobotics2425laptop11/pyvhr_ros_env/bin/python"
    venv_python = "/home/mscrobotics2425laptop11/rppg_env/bin/python"

    # Path to your actual ROS 2 node
    # node_script = os.path.expanduser("~/ws_dis/src/rppg_node/rppg_node/rppg_node.py")
    node_script = os.path.expanduser("~/ws_dis/src/rppg_node/rppg_node/rppg_toolbox_nn_node.py")

    # Run the script using the virtualenv Python
    subprocess.run([venv_python, node_script])
