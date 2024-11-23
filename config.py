import os

class Config:
    UPLOAD_FOLDER = 'static/uploads/'
    MAX_CONTENT_LENGTH = 1024 * 1024  # 限制檔案大小為 1MB
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # MODEL_PATH = 'trained_model.keras' # emotion_model_utils 需要的檔案路徑
    MODEL_PATH = 'trained_model.tflite' # emotion_model_utils 需要的檔案路徑
    
    CASCADE_PATH = 'Opencv_Images/haarcascade_frontalface_default.xml' # face_recognition_model_utils 需要的檔案路徑
    FACE_MODEL_PATH = 'Opencv_Images/face.yml'
