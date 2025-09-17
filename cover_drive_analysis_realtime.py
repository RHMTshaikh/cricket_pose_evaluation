from models import Model_Module_Interface, get_model

model_name = 'mediapipe' # or 'openpose'
model: Model_Module_Interface = get_model(model_name=model_name)

video1 = r'input/Stunning_cover_drive.mp4'  
model.process_video(video1)