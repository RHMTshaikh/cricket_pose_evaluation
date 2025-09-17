import cv2 as cv
import numpy as np


def calculate_front_elbow_angle(scaled_landmarks, frame=None):
    """
    Calculate the front elbow angle using the landmarks.
    Assumes landmarks is a dictionary with keys 'right_shoulder', 'right_elbow', and 'right_wrist'.
    Each key maps to a tuple (x, y).
    """
    if not scaled_landmarks:
        raise ValueError("Landmarks data is required to calculate the elbow angle.")
    
    shoulder_r = scaled_landmarks[12]
    shoulder_l = scaled_landmarks[11]
    
    if shoulder_r.z < shoulder_l.z:
        shoulder = scaled_landmarks[12]
        elbow = scaled_landmarks[14]
        wrist = scaled_landmarks[16]
    else:
        shoulder = scaled_landmarks[11]
        elbow = scaled_landmarks[13]
        wrist = scaled_landmarks[15]
        
    if frame is not None:
        _draw_landmarks_on_image(frame, (shoulder, elbow, wrist))

    angle = _angle_between_landmarks(shoulder, elbow, wrist)
    return angle

def calculate_spine_lean_angle(scaled_landmarks, frame=None):
    """
    Calculate the spine lean angle using the landmarks.
    Assumes landmarks is a dictionary with keys 'right_hip', 'right_shoulder', and 'right_ear'.
    Each key maps to a tuple (x, y).
    """
    if not scaled_landmarks:
        raise ValueError("Landmarks data is required to calculate the spine lean angle.")
    
    hip_r = scaled_landmarks[24]
    hip_l = scaled_landmarks[23]
    shoulder_r = scaled_landmarks[12]
    shoulder_l = scaled_landmarks[11]
    
    hip_r = np.array([hip_r.x, hip_r.y, hip_r.z])
    hip_l = np.array([hip_l.x, hip_l.y, hip_l.z])
    shoulder_r = np.array([shoulder_r.x, shoulder_r.y, shoulder_r.z])
    shoulder_l = np.array([shoulder_l.x, shoulder_l.y, shoulder_l.z])

    hip = (hip_r + hip_l) / 2
    shoulder = (shoulder_r + shoulder_l) / 2
    
    spine_vector = shoulder - hip
    vertical_vector = np.array([0, -1, 0])  # y-axis points down

    angle = _angle_between_lines(spine_vector, vertical_vector)
    if frame is not None:
        _draw_landmarks_on_image(frame, (hip_r, hip_l, shoulder_r, shoulder_l))
    return angle

def calculate_head_over_knee_angle(scaled_landmarks, frame=None):
    if not scaled_landmarks:
        raise ValueError("Landmarks data is required to calculate the spine lean angle.")
    
    nose = scaled_landmarks[0]
    ear_r = scaled_landmarks[8]
    ear_l = scaled_landmarks[7]
    shoulder_l = scaled_landmarks[11]

    nose = np.array([nose.x, nose.y, nose.z])
    ear_r = np.array([ear_r.x, ear_r.y, ear_r.z])
    ear_l = np.array([ear_l.x, ear_l.y, ear_l.z])

    head = (nose + ear_r + ear_l) / 3
    
    knee_r = scaled_landmarks[26]
    knee_l = scaled_landmarks[25]
    
    if knee_r.z < knee_l.z:
        front_knee = scaled_landmarks[26]
    else:
        front_knee = scaled_landmarks[25]
    if frame is not None:
        _draw_landmarks_on_image(frame, (front_knee,))
    
    front_knee = np.array([front_knee.x, front_knee.y, front_knee.z])
    vertical_vector = np.array([0, -1, 0])  

    line = head - front_knee

    angle = _angle_between_lines(line, vertical_vector)
    return angle

def calculate_front_foot_angle(scaled_landmarks, frame=None):
    if not scaled_landmarks:
        raise ValueError("Landmarks data is required to calculate the front foot angle.")
    
    ankle_r = scaled_landmarks[28]
    ankle_l = scaled_landmarks[27]
    
    if ankle_r.z < ankle_l.z:
        front_ankle = scaled_landmarks[28]
        front_foot_index = scaled_landmarks[32]
        crease_line = np.array([1, 0, 0])  # x-axis
    else:
        front_ankle = scaled_landmarks[27]
        front_foot_index = scaled_landmarks[31]
        crease_line = np.array([-1, 0, 0])  # x-axis
        
    if frame is not None:
        _draw_landmarks_on_image(frame, (front_ankle, front_foot_index))
        
    foot_line = np.array([front_foot_index.x - front_ankle.x, front_foot_index.y - front_ankle.y, front_foot_index.z - front_ankle.z])

    angle = _angle_between_lines(foot_line, crease_line)
    return angle


def _draw_landmarks_on_image(frame, landmarks: tuple):
    """
    Draws the landmarks and connections on the image.
    """
    for landmark in landmarks:
        cv.circle(frame, (landmark.x, landmark.y), 5, (0, 255, 0), -1)
    
def _distance_between_landmark(landmarkA, landmarkB):
    """
    Calculate the Euclidean distance between two landmarks.
    Each landmark is expected to have 'x' and 'y' attributes normalized between 0 and 1.
    """
    AB = np.array([landmarkA.x - landmarkB.x, landmarkA.y - landmarkB.y, landmarkA.z - landmarkB.z])
    return np.linalg.norm(AB)

def _angle_between_landmarks(landmarkA, landmarkB, landmarkC):
    """
    Calculate the angle at landmarkB formed by the line segments landmarkA-landmarkB and landmarkC-landmarkB.
    """
    BA = np.array([landmarkA.x - landmarkB.x, landmarkA.y - landmarkB.y, landmarkA.z - landmarkB.z])
    BC = np.array([landmarkC.x - landmarkB.x, landmarkC.y - landmarkB.y, landmarkC.z - landmarkB.z])
    
    angle_degrees = _angle_between_lines(BA, BC)
    
    return angle_degrees

def _angle_between_lines(lineA, lineB):
    """Calculate the angle between two lines defined by two points each."""

    cos = np.dot(lineA, lineB) / (np.linalg.norm(lineA) * np.linalg.norm(lineB))

    angle = np.arccos(cos)
    angle_degrees = np.degrees(angle)
    
    return angle_degrees

class BiomechanicalConstants:
    front_elbow_angle_ideal = 90
    spine_lean_angle_ideal = 0
    head_over_knee_angle_ideal = 0
    front_foot_angle_ideal = 90

    front_elbow_angle_threshold = 15
    spine_lean_angle_threshold = 10
    head_over_knee_angle_threshold = 10
    front_foot_angle_threshold = 15