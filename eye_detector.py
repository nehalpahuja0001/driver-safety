import cv2
import numpy as np
from scipy.spatial import distance
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

def calculate_ear(landmarks, eye_indices):
    pts = [(landmarks[i].x, landmarks[i].y) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C)

# FaceMesh model download karo pehli baar
import urllib.request, os
MODEL = "face_landmarker.task"
if not os.path.exists(MODEL):
    print("Model download ho raha hai...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL
    )
    print("Model ready!")

base_options = python.BaseOptions(model_asset_path=MODEL)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
landmarker = vision.FaceLandmarker.create_from_options(options)

def get_ear_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, frame
    lms = result.face_landmarks[0]
    left  = calculate_ear(lms, LEFT_EYE)
    right = calculate_ear(lms, RIGHT_EYE)
    ear = (left + right) / 2.0
    return round(ear, 3), frame
