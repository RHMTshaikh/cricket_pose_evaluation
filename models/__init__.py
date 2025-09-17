import os
import importlib
from abc import ABC, abstractmethod
import cv2 as cv
import time
import json

from biomechanical_metrics import (calculate_front_elbow_angle, 
                                   calculate_spine_lean_angle, 
                                   calculate_head_over_knee_angle, 
                                   calculate_front_foot_angle, 
                                   BiomechanicalConstants)
from biomechanical_metrics.speed import SpeedCalculator
from biomechanical_metrics.scores import calculate_biomechanical_score


class Scaled_Landmark:
    def __init__(self):
        self.x = None
        self.y = None
        self.z = None
        self.visibility = None
        
class Model_Module_Interface(ABC):
    

    @abstractmethod
    def get_landmarks(self, frame):
        raise NotImplementedError("Subclasses should implement this method.")
    
    @abstractmethod
    def draw_landmarks(self, frame, landmarks):
        raise NotImplementedError("Subclasses should implement this method.")
    

    def draw_text_with_bg(self, img, text, pos, font_scale=0.6, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
        font = cv.FONT_HERSHEY_SIMPLEX
        thickness = 2
        
        padding = 5  # Equal padding on all sides
        (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)

        rect_width = text_width + 2 * padding
        rect_height = text_height + baseline + 2 * padding
        
        x, y = pos  # Calculate rectangle position
        rect_x1 = x
        rect_y1 = y 
        rect_x2 = rect_x1 + rect_width
        rect_y2 = rect_y1 + rect_height
        
        cv.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)  # Draw background rectangle
        
        text_x = rect_x1 + padding  # Center text in rectangle with equal padding
        text_y = rect_y1 + padding + text_height  # Properly centered vertically
        
        cv.putText(img, text, (text_x, text_y), font, font_scale, text_color, thickness)

    def get_scaled_landmarks(self, landmarks, W, H):
        scaled_landmarks = []
        for landmark in landmarks.landmark:
            scaled_landmark = Scaled_Landmark()
            scaled_landmark.x = int(landmark.x * W)
            scaled_landmark.y = int(landmark.y * H)
            scaled_landmark.z = int(landmark.z * W)
            scaled_landmark.visibility = landmark.visibility
            
            scaled_landmarks.append(scaled_landmark)
        
        return scaled_landmarks
    
    def process_video(self, video_path):
        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return
        
        W, H, fps = (int(cap.get(x)) for x in [cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_FPS])
        scale = 800 / max(H, W)
        H_scaled, W_scaled = int(H*scale), int(W*scale)
        video_writer = cv.VideoWriter('output/annotated_video.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, (W_scaled, H_scaled))
        avg_fps = []

        skip = 1
        frame_count = 0
        green = (0, 255, 0)
        red = (0, 0, 255)
        white = (255, 255, 255)
        black = (0, 0, 0)
        
        wrist_speed_calculator = SpeedCalculator()
        
        striking = False
        max_strike_speed = 0
        
        evaluation_json  = {}
        
        front_elbow_angle_ideal    = BiomechanicalConstants.front_elbow_angle_ideal
        spine_lean_angle_ideal     = BiomechanicalConstants.spine_lean_angle_ideal
        head_over_knee_angle_ideal = BiomechanicalConstants.head_over_knee_angle_ideal
        front_foot_angle_ideal     = BiomechanicalConstants.front_foot_angle_ideal

        front_elbow_angle_threshold    = BiomechanicalConstants.front_elbow_angle_threshold
        spine_lean_angle_threshold     = BiomechanicalConstants.spine_lean_angle_threshold
        head_over_knee_angle_threshold = BiomechanicalConstants.head_over_knee_angle_threshold
        front_foot_angle_threshold     = BiomechanicalConstants.front_foot_angle_threshold

        front_elbow_angle_score = 0
        spine_lean_angle_score = 0
        head_over_knee_angle_score = 0
        front_foot_angle_score = 0
        
        strike_frame_count = 0
        
        while True:
            frame_count += 1
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip != 0:
                continue

            frame = cv.resize(frame, (W_scaled, H_scaled))

            landmarks = self.get_landmarks(frame)
            annotated_frame = self.draw_landmarks(frame, landmarks)
            # time.sleep(0.3)  # Simulate processing delay
            
            if landmarks:
                scaled_landmarks = self.get_scaled_landmarks(landmarks, W_scaled, H_scaled)
                
                SpeedCalculator.set_scale_factor(scaled_landmarks)

                wrist = scaled_landmarks[self.keypoints.RIGHT_WRIST]  # Right wrist
                wrist_speed = wrist_speed_calculator.calculate_speed((wrist.x, wrist.y), frame_count)
                max_strike_speed = max(max_strike_speed, wrist_speed)

                front_elbow_angle    = calculate_front_elbow_angle(scaled_landmarks)
                spine_lean_angle     = calculate_spine_lean_angle(scaled_landmarks)
                head_over_knee_angle = calculate_head_over_knee_angle(scaled_landmarks)
                front_foot_angle     = calculate_front_foot_angle(scaled_landmarks)

                if not striking and wrist_speed > 0.6:
                    striking = True
                if striking and wrist_speed < 0.1:
                    striking = False
                    
                if striking:
                    strike_frame_count += 1
                    
                    front_elbow_angle_score    = calculate_biomechanical_score( strike_frame_count, front_elbow_angle_score,    front_elbow_angle,    front_elbow_angle_ideal,    front_elbow_angle_threshold    )
                    spine_lean_angle_score     = calculate_biomechanical_score( strike_frame_count, spine_lean_angle_score,     spine_lean_angle,     spine_lean_angle_ideal,     spine_lean_angle_threshold     )
                    head_over_knee_angle_score = calculate_biomechanical_score( strike_frame_count, head_over_knee_angle_score, head_over_knee_angle, head_over_knee_angle_ideal, head_over_knee_angle_threshold )
                    front_foot_angle_score     = calculate_biomechanical_score( strike_frame_count, front_foot_angle_score,     front_foot_angle,     front_foot_angle_ideal,     front_foot_angle_threshold     )

                    evaluation_json ['front_elbow_angle_score']    = front_elbow_angle_score
                    evaluation_json ['spine_lean_angle_score']     = spine_lean_angle_score
                    evaluation_json ['head_over_knee_angle_score'] = head_over_knee_angle_score
                    evaluation_json ['front_foot_angle_score']     = front_foot_angle_score

                    self.draw_text_with_bg(annotated_frame, 'STRIKING', (0, 350), font_scale=0.7, text_color=red, bg_color=white)
                    
                front_elbow_angle_color    = green if front_elbow_angle > 75 and front_elbow_angle < 115 else red
                spine_lean_angle_color     = green if spine_lean_angle < 10                              else red
                head_over_knee_angle_color = green if head_over_knee_angle < 10                          else red
                front_foot_angle_color     = green if front_foot_angle > 75 and front_foot_angle < 115   else red

                self.draw_text_with_bg(annotated_frame, f'Elbow: {int(front_elbow_angle)} deg',        (0, 385), font_scale=0.7, text_color=front_elbow_angle_color,    bg_color=white)
                self.draw_text_with_bg(annotated_frame, f'Spine: {int(spine_lean_angle)} deg',         (0, 420), font_scale=0.7, text_color=spine_lean_angle_color,     bg_color=white)
                self.draw_text_with_bg(annotated_frame, f'Head-Knee: {int(head_over_knee_angle)} deg', (0, 455), font_scale=0.7, text_color=head_over_knee_angle_color, bg_color=white)
                self.draw_text_with_bg(annotated_frame, f'Foot: {int(front_foot_angle)} deg',          (0, 490), font_scale=0.7, text_color=front_foot_angle_color,     bg_color=white)
                self.draw_text_with_bg(annotated_frame, f'Wrist Speed: {wrist_speed:.2f} m/s',         (0, 525), font_scale=0.7, text_color=black,                      bg_color=white)

                del scaled_landmarks

            process_time = time.time() - start_time
            
            fps = 1 / process_time if process_time > 0 else 0
            avg_fps.append(fps)
            fps = sum(avg_fps) / len(avg_fps)
            font_scale = 0.6
            self.draw_text_with_bg(annotated_frame, f"FPS: {fps:.2f}", (0, 0), font_scale=font_scale, bg_color=(0, 0, 100))
            self.draw_text_with_bg(annotated_frame, f"Process Time: {process_time*1000:.2f} ms", (0, 35), font_scale=font_scale, bg_color=(0, 100, 0))

            video_writer.write(annotated_frame)

            cv.imshow("Video", annotated_frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        video_writer.release()
        cv.destroyAllWindows()
        
        evaluation_json ['front_elbow_angle']     = 'good' if front_elbow_angle_score > 75 else 'needs improvement'
        evaluation_json ['spine_lean_angle']      = 'good' if spine_lean_angle_score > 75 else 'needs improvement'
        evaluation_json ['head_over_knee_angle']  = 'good' if head_over_knee_angle_score > 75 else 'needs improvement'
        evaluation_json ['front_foot_angle']      = 'good' if front_foot_angle_score > 75 else 'needs improvement'

        evaluation_json ['max_strike_wrist_speed'] = f"{round(max_strike_speed, 2)} m/s"

        filename = "output/evaluation.json"
        with open(filename, 'w') as json_file:
            json.dump(evaluation_json , json_file, indent=4)
        json_file.close()
        


models_dir = os.path.dirname(__file__)

available_models = {}
for filename in os.listdir(models_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        model_name = filename[:-3]
        module = importlib.import_module(f'.{model_name}', package=__package__)
        available_models[model_name] = module.model
        
def get_model(model_name):
    if model_name not in available_models:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(available_models.keys())}")
    return available_models[model_name]