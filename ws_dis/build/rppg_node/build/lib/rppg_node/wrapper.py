import subprocess
import os
import sys

def main():
    # Path to virtualenv Python
    venv_python = "/home/mscrobotics2425laptop11/rppg_env/bin/python"

    # Path to ROS 2 node
    node_script = os.path.expanduser("~/ws_dis/src/rppg_node/rppg_node/webcam_reader.py")

    # Run the script using the virtualenv Python
    subprocess.run([venv_python, node_script])
