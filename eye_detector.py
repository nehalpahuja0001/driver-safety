import cv2
import numpy as np
import math
from scipy.spatial import distance
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# 3D model points for head pose estimation
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
])

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    A = distance.euclidean(pts[1], pts[5])
    B = distance.euclidean(pts[2], pts[4])
    C = distance.euclidean(pts[0], pts[3])
    if C == 0:
        return 0.0
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

def get_face_features(frame):
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None, None, frame

    lms = result.face_landmarks[0]
    
    # Calculate EAR
    left  = calculate_ear(lms, LEFT_EYE, w, h)
    right = calculate_ear(lms, RIGHT_EYE, w, h)
    ear = (left + right) / 2.0
    
    # Head Pose Estimation
    nose_tip = (lms[1].x * w, lms[1].y * h)
    chin = (lms[152].x * w, lms[152].y * h)
    left_eye_corner = (lms[33].x * w, lms[33].y * h)
    right_eye_corner = (lms[263].x * w, lms[263].y * h)
    left_mouth = (lms[61].x * w, lms[61].y * h)
    right_mouth = (lms[291].x * w, lms[291].y * h)
    
    image_points = np.array([
        nose_tip, chin, left_eye_corner, right_eye_corner, left_mouth, right_mouth
    ], dtype="double")
    
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    pitch = 0.0
    if success:
        rmat, _ = cv2.Rodrigues(rotation_vector)
        sy = np.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rmat[2,1], rmat[2,2])
        else:
            x = math.atan2(-rmat[1,2], rmat[1,1])
        pitch = x * 180.0 / math.pi
        
        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180

    return round(ear, 3), pitch, frame
