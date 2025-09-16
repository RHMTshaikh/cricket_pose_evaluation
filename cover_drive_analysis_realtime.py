import models

model_name = 'mediapipe'
model = models.get_model(model_name=model_name)

video1 = r'video/file/path.mp4'  
model.process_video(video1)