# rPPG_ROS: Modular ROS2 Pipeline for Camera-Based Physiological Sensing

This repository provides a ROS2-integrated pipeline for remote photoplethysmography (rPPG). It supports video-based inference, real-time webcam based inference and modular deployment via Docker.

---

## Features

- ROS2-compatible rPPG pipeline with modular node structure
- Integration of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) for camera-based vitals estimation
- Real-time inference using webcamera
- Uses Mediapipe Face detection algorithm for Region of Interest selection (RoI)
- Dockerized deployment with reproducible environment setup

---

## Included Dependencies

This repository includes [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) as a Git submodule under the Responsible AI Source Code License v1.1.
Please review the full license [here](https://github.com/ubicomplab/rPPG-Toolbox/blob/main/LICENSE) or at [http://licenses.ai](http://licenses.ai).

By using this repository, you agree not to use rPPG-Toolbox or any derivative work for:
- Surveillance or identity inference
- Healthcare prediction or insurance decisions
- Criminal profiling or prediction

These restrictions are enforceable and must be passed on to any downstream users.

## Repository Structure

```text
rPPG_ROS/
├─ rppg_ws
    ├─ build
    ├─ install
    ├─ log
    ├─ src/rppg_node
        ├─ config
            ├─ rppg_run_params.yaml
        ├─ launch
            ├─ rppg_pipeline_video.launch.py
            ├─ rppg_pipeline.launch.py
        ├─ resource
        ├─ rppg_node
            ├─ __init__.py
            ├─ rppg_toolbox_node_run.py
            ├─ rppg_toolbox_video_node.py
            ├─ webcam_reader.py
        ├─ test
        ├─ utils
            ├─ __init__.py
            ├─ face_roi_detection.py
            ├─ utils.py
        package.xml
        setup.cfg
        setup.py
├─ rppgtb
Dockerfile_Humble
Dockerfile_Jazzy
requirements.txt
```

## Configuration File: (`rppg_urn_params.yaml`)

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
| `img_height`    | `int`    | `128`                    | RoI height; resized to 72×36 for DeepPhys/EfficientPhys. |
| `img_width`     | `int`    | `128`                    | RoI width; resized to 72×36 for DeepPhys/EfficientPhys. |
| `roi_area`      | `str`    | `'all'`                  | Region of Interest. Options: `'forehead'`, `'left cheek'`, `'right cheek'`, `'all'`. |
| `viz`           | `bool`   | `true`                   | Enable visualization (e.g. `cv2.imshow()`). |


## Installation

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/srivatsan3/rPPG_ROS.git
```
This clones both the `rppg_ws` Workspace and the `rppgtb` rPPG-Toolbox

### 2.a For Docker Based approach
Use `Dockerfile_Humble` for ROS2 Humble Version (supported by Ubuntu 22.04)<br>
Use `Dockerfile_Jazzy` for ROS2 Jazzy Version (Supported by Ubuntu 24.04)<br>

Please rename whichever file you are using as simply `Dockerfile` as that is what docker commands read from.<br>

Once inside the `rPPG_ROS` folder, use the following commands to build and run the dockerfile.<br>

```bash
docker run -it   --name rppg_container   --rm   --net=host   --privileged   rppg_container`
xhost +local:docker`
docker run -it   --name rppg_container   --rm   --net=host   --privileged   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   rppg_container`
```


This installs the ROS2 images, creates a python3.10 virtual environment (`venv`), installs essential python libraries.It also copies the entire codebase to Docker space. 

### 2.b If your laptop already has ROS2 installed, you can use this:
Create a python venv using: 
`python3 -m venv ~/venv` and activate it using `source ~/venv/bin/activate`

Go inside the folder `rPPG_ROS`. <br>
Use the `requirements.txt` to perform `pip install -r requirements.txt`.<br>
Many of the requirements are ROS2 related which should be available automatically. <br>
The key python libraries in this requirements file are <br>
`scipy mediapipe opencv_python pyyaml mat73 scikit-learn scikit-image torch cv_bridge`

Once the libraries are installed, you can move to next step.

### 3. Changes to be made
Use the `config/rppg_run_params.yaml` to modify these file paths aligning with your folder structures:
1) `rpptb_path`: Which must be `<prefix>/rPPG_ROS/rppgtb/rPPG-Toolbox` in default setting
2) `python_path`: Which must be `~/venv/bin/python` in default setting

### 4. Running the Code
Build and Source the package (Make sure you are in the rppg_ws folder, if not use `cd`)<br>
```bash
colcon build
source install/setup.bash
```

Make use of the correct launch file:
`ros2 launch rppg_node rppg_pipeline.launch.py` for the webcam based pipeline
`ros2 launch rppg_node rppg_pipeline_video.launch.py` for inferencing using an existing video (Make sure the path is valid)

## Common Errors:
1) One of the Errors that can occur while running `deepphys` or `efficientphys` can be `.view` function related. If encountered, edit the following:<br>
Go to `rPPG_ROS/rppgtn/rPPG-Toolbox/neural_methods/model/`<br>
    i. Replace `d9 = d8.view(d8.size(0), -1)` with `d9 = d8.reshape(d8.size(0), -1)` in `DeepPhys.py` while running `Deep Phys` <br>
    ii. Replace `d9 = d8.view(d8.size(0), -1)` with `d9 = d8.reshape(d8.size(0), -1)` in `EfficientPhys.py` while running `Efficient Phys` <br>
2) If docker command does not work showing any permission errors use `sudo` in the beginning of the command. <br>
3) Make sure paths of rppg tool box, python interpretor, video files (if any) are correct.

## Citation: rPPG-Toolbox

This repository integrates [`rPPG-Toolbox`](https://github.com/ubicomplab/rPPG-Toolbox), a deep learning framework for remote photoplethysmography developed by the Ubicomp Lab at the University of Washington.

If you use this, then as you are using rPPG-Toolbox in your work, please cite the following paper: <br>

```text
@article{liu2022rppg,
  title={rPPG-Toolbox: Deep Remote PPG Toolbox},
  author={Liu, Xin and Narayanswamy, Girish and Paruchuri, Akshay and Zhang, Xiaoyu and Tang, Jiankai and Zhang, Yuzhe and Wang, Yuntao and Sengupta, Soumyadip and Patel, Shwetak and McDuff, Daniel},
  journal={arXiv preprint arXiv:2210.00716},
  year={2022}
}
```

GitHub repository: [https://github.com/ubicomplab/rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
