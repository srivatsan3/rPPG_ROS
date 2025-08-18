import numpy as np
import mat73
import cv2


# def read_video(video_path, dict_key = None):
#     if video_path.split('.')[-1] == 'mat':
#         data = mat73.loadmat(video_path)
#         if type(data) == dict:
#             frames = data[dict_key] 
#             frame = frames[0]
#             # print(f'{frame.dtype}, Min: {frame.min()}, Max: {frame.max()}')
#             frames_rgb = [cv2.cvtColor(np.clip(frame*255,0,255).astype(np.uint8),cv2.COLOR_RGB2BGR) for frame in frames]
#             frame_rgb = frames_rgb[0]
#             # print(f'{frame_rgb.dtype}, Min: {frame_rgb.min()}, Max: {frame_rgb.max()}')
 
#             video_frames = np.array(frames_rgb)
#             # print(video_frames.shape)
#             # raise NameError
#     else:
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             raise IOError('Cannot Open Video File:',video_path)
        
#         video_frames = []
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             video_frames.append(frame)
#         # video_frames = np.array(video_frames)
#         cap.release()
#     return video_frames

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

            