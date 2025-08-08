import cv2
import mediapipe as mp
import contextlib
from mediapipe.framework.formats import landmark_pb2

FOREHEAD_LANDMARKS = [54,103,67,109,10, 338, 297, 332, 284, 333,299,337,151,108,69,104,68]
LEFT_CHEEK_LANDMARKS = [280,346,347,330,266,425,411]
RIGHT_CHEEK_LANDMARKS = [50,123,187,205,36,101,118,117]

mp_face_mesh = mp.solutions.face_mesh


def extract_face_roi(frame, box_size=128, face_detection = None):
    '''
    Function to extract the region of interest (Face)

    Parameters:
    frame : An array representing a single frame of the video
    box_size: The size of region of interest required
    face_detection : Face Dectection method

    Returns:
    face_crop : Cropped image with detected face in it
    detection : Key points detection and boundix box information
    '''
    if face_detection is None:
        return None
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.detections:
        return None,None  # No face detected

    # Use first detected face
    detection = results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = frame.shape

    x1 = int(bbox.xmin * w)
    y1 = int(bbox.ymin * h)
    x2 = int((bbox.xmin + bbox.width) * w)
    y2 = int((bbox.ymin + bbox.height) * h)

    face_crop = frame[y1:y2, x1:x2]
    face_crop = cv2.resize(face_crop, (box_size, box_size))

    return face_crop, detection


def extract_face_regions(frame, roi = 'FOREHEAD', target_size=(128,128)):
    
    with contextlib.redirect_stderr(None):
        with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                    refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if not results.multi_face_landmarks:
                return None,None

        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark
        
        if roi == 'FOREHEAD':
            roi_lm = FOREHEAD_LANDMARKS
        elif roi == 'LEFT CHEEK':
            roi_lm = LEFT_CHEEK_LANDMARKS
        elif roi == 'RIGHT_CHEEK':
            roi_lm =  RIGHT_CHEEK_LANDMARKS


        full_landmarks = results.multi_face_landmarks[0].landmark
        roi_landmarks = landmark_pb2.NormalizedLandmarkList(
            landmark=[full_landmarks[i] for i in roi_lm]
        )
        # print(roi_landmarks[54],roi_lm)
        points = [landmarks[i] for i in roi_lm]

        # print('**************',points)
        xs = [int(p.x * w) for p in points]
        ys = [int(p.y * h) for p in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        # print('**************',x1,x2,y1,y2)
        roi_crop = frame[y1:y2, x1:x2]
        
        if roi_crop.size == 0:
            return None, None
        final_img =  cv2.resize(roi_crop, target_size)

        return final_img, roi_landmarks
    


