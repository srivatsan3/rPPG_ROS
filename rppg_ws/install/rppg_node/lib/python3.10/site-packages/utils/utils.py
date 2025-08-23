import numpy as np
import mat73
import cv2

def read_video_stream(video_path, dict_key=None):
    '''
    Reads frames from a video file or a .mat file containing video data.
    If a .mat file is provided, it expects the video frames to be stored under the specified dict_key (In support of SCAMPS dataset format).
    If a video file is provided, it reads frames using OpenCV.

    Parameters:
    video_path (str): Path to the video file or .mat file.
    dict_key (str): Key to access video frames in the .mat file. If None, it assumes the video file is in a standard format.

    Returns:
    Generator yielding frames as numpy arrays.
    '''

    if video_path.endswith('.mat'):                                 # If the video is in a .mat file
        data = mat73.loadmat(video_path)                            # Load the .mat file
        frames = data[dict_key]
        for frame in frames:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)   # Convert frame to uint8 format
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)            # Convert RGB to BGR for OpenCV compatibility
    else:
        cap = cv2.VideoCapture(video_path)                          # Open the video file using OpenCV  
        try:
            while cap.isOpened():                                   # Check if the video capture is opened   
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame                                         # Yield the frame                                 
        finally:
            cap.release()

            