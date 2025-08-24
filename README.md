# rPPG_ROS: Modular ROS2 Pipeline for Camera-Based Physiological Sensing

This repository provides a ROS2-integrated pipeline for remote photoplethysmography (rPPG). It supports video-based inference, real-time webcam based inference and modular deployment via Docker.

---

## Features

- ROS2-compatible rPPG pipeline with modular node structure
- Integration of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) for camera-based vitals estimation
- Real-time inference using webcamera
- Dockerized deployment with reproducible environment setup

---

## Configuration File: (`rppg_run_params.yaml`)

| Parameter       | Type     | Default / Example Value | Description |
|----------------|----------|--------------------------|-------------|
| `webcam_id`     | `int`    | `0`                      | Webcam ID used for live capture (only for webcam-based runs). |
| `frame_rate`    | `int`    | `30`                     | Frame rate of the input source; should match webcam FPS if used. |
| `algo`          | `str`    | `'pos'`        | Algorithm to use. Supported: `'chrom'`, `'pos'`, `'green'`, `'ica'`, `'pbv'`, `'pbv2'`, `'lgi'`, `'omit'`, `'physnet'`, `'efficientphys'`, `'deepphys'`, `'bigsmall'`. |
| `estimate`      | `str`    | `'fft'`                  | BPM estimation method. Supported: `'fft'`, `'peak'`. |
| `window_secs`   | `int`    | `8`                      | Duration of each processing window in seconds. |
| `overlap_secs`  | `int`    | `6`                      | Overlap between consecutive windows in seconds. |
| `camera_topic`  | `str`    | `'/camera'`              | ROS topic for camera frames (ignored for video-based runs). |
| `topic`         | `str`    | `'/heart_rate_bpm'`      | ROS topic to publish estimated heart rate. |
| `rppgtb_path`   | `str`    | `/home/.../rPPG-Toolbox` | Absolute path to the rPPG Toolbox directory. |
| `video_path`    | `str`    | `/home/.../vid.avi`      | Path to input video file (used only for video-based runs). |
| `python_path`   | `str`    | `/home/.../python`       | Path to Python interpreter (e.g. from virtual environment). |
| `img_height`    | `int`    | `128`                    | Input image height; resized to 36×36 for DeepPhys/EfficientPhys. |
| `img_width`     | `int`    | `128`                    | Input image width; resized to 36×36 for DeepPhys/EfficientPhys. |
| `roi_area`      | `str`    | `'all'`                  | Region of interest. Options: `'forehead'`, `'left cheek'`, `'right cheek'`, `'all'`. |
| `viz`           | `bool`   | `true`                   | Enable visualization (e.g. `cv2.imshow()`). |


## Installation

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/srivatsan3/rPPG_ROS.git
```
This clones both the `rppg_ws` Workspace and the `rppgtb` rPPG-Toolbox

### 2.a For Docker Based approach
Use `Dockerfile_Humble` for ROS2 Humble Version (supported by Ubuntu 22.04)
Use `Dockerfile_Jazzy` for ROS2 Jazzy Version (Supported by Ubuntu 24.04)

Please rename whichever file you are using as simply `Dockerfile` as that is what docker commands read from.

Once inside the `rPPG_ROS` folder, use the following commands to build and run the dockerfile.

```bash
docker run -it   --name rppg_container   --rm   --net=host   --privileged   rppg_container`
xhost +local:docker`
docker run -it   --name rppg_container   --rm   --net=host   --privileged   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   rppg_container`
```

If docker command does not work showing any permission errors use `sudo` in the beginning of the command.

This installs the ROS2 images, creates a python3.10 virtual environment (`venv`), installs essential python libraries.
It also copies the entire codebase to Docker space. 

### 2.b If your laptop already has ROS2 installed, you can use this:
Create a python venv using: 
`python3 -m venv ~/venv` and activate it using `source ~/venv/bin/activate`

Go inside the foler `rPPG_ROS`
Use the `requirements.txt` to perform `pip install -r requirements.txt`
Many of the requirements are ROS2 related which should be available automatically. 
The key python libraries can be installed by `pip install scipy mediapipe opencv_python pyyaml mat73 scikit-learn scikit-image torch psutil objgraph`

Once the libraries are installed, you can move to next step.

### 3. Changes to be made
Use the `config/rppg_run_params.yaml` to modify these file paths aligning with your folder structures:
1) rpptb_path: Which must be `<prefix>/rPPG_ROS/rppgtb/rPPG-Toolbox` in default setting
2) python_path: Which must be `~/venv/bin/python` in default setting

### 4. Running the Code
Build and Source the package (Make sure you are in the rppg_ws folder, if not use `cd`)
`colcon build`
`source install/setup.bash`

Make use of the correct launch file:
`ros2 launch rppg_node rppg_pipeline.launch.py` for the webcam based pipeline
`ros2 launch rppg_node rppg_pipeline_video.launch.py` for inferencing using an existing video (Make sure the path is valid)


