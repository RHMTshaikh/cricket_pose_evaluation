import numpy as np

class SpeedCalculator:
    assumed_shoulder_width_m = 0.40
    scale_factor = None
    
    def __init__(self):
        self.previous_frame_count = None
        self.previous_position = None
        self.speed = None
        self.ALPHA = 0.9  # Smoothing factor for speed
    
    @classmethod
    def set_scale_factor(cls, scaled_landmarks):
        if not scaled_landmarks:
            raise ValueError("Scaled landmarks are required to set the scale factor.")

        # Calculate the scale factor based on the assumed shoulder width
        shoulder_width = np.linalg.norm(np.array([scaled_landmarks[12].x, scaled_landmarks[12].y]) - np.array([scaled_landmarks[11].x, scaled_landmarks[11].y]))
        if shoulder_width == 0:
            return
        confidence = 0.2
        
        if cls.scale_factor is None:
            cls.scale_factor = cls.assumed_shoulder_width_m / shoulder_width
        else:
            new_scale_factor = cls.assumed_shoulder_width_m / shoulder_width
            cls.scale_factor = (confidence * new_scale_factor) + ((1 - confidence) * cls.scale_factor)
            
    def calculate_speed(self, current_position, frame_count):
        if self.__class__.scale_factor is None:
            raise ValueError("Scale factor is not set. Call set_scale_factor() first.")
        
        if self.previous_frame_count is None or self.previous_position is None:
            self.previous_frame_count = frame_count
            self.previous_position = current_position
            return 0.0  # No speed can be calculated on the first call

        time_diff = frame_count - self.previous_frame_count
        if time_diff <= 0:
            return 0.0  # Avoid division by zero or negative time intervals

        distance = np.linalg.norm(np.array(current_position) - np.array(self.previous_position))
        speed = distance / time_diff * self.__class__.scale_factor  # Speed in units per second

        self.previous_frame_count = frame_count
        self.previous_position = current_position
        
        if self.speed is None:
            self.speed = speed
        else:
            self.speed = self.ALPHA * speed + (1 - self.ALPHA) * self.speed

        return self.speed # Speed in m/s