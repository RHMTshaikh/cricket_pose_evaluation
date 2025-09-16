import mediapipe as mp

from models import Model_Module_Interface

class MediapipeModel(Model_Module_Interface):
    def __init__(self):
        self.model = mp.solutions.pose.Pose(min_detection_confidence=0.7,
                                            min_tracking_confidence=0.3,
                                            model_complexity=2,
                                            smooth_landmarks=True)

        self.keypoints = mp.solutions.pose.PoseLandmark

    def get_landmarks(self, frame):
        results = self.model.process(frame)
        return results.pose_landmarks

    def draw_landmarks(self, frame, landmarks):
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
        )
        return frame

model = MediapipeModel()
