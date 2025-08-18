import numpy as np
import mat73
import cv2

def read_video_stream(video_path, dict_key=None):
    if video_path.endswith('.mat'):
        data = mat73.loadmat(video_path)
        frames = data[dict_key]
        for frame in frames:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            yield cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        cap = cv2.VideoCapture(video_path)
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

            