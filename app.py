from flask import Flask
from routes.index import index_bp
from routes.upload import upload_bp
from routes.video_feed import video_feed_bp
from routes.video_feed_from_client import video_feed_from_client_bp
from config import Config
from utils.face_recognition_model_utils import initialize_faceRecognition_resources
from utils.emotion_model_utils import initialize_emotion_model
# from utils.emotion_model_utils_for_tf import initialize_emotion_model
import os

app = Flask(__name__)
app.config.from_object(Config)

# 在 Flask 應用啟動時初始化全域資源 
@app.before_request
def initialize_resources():
    if not hasattr(app, 'resources_initialized'):
        initialize_faceRecognition_resources(app)
        initialize_emotion_model(app)
        app.resources_initialized = True

# 註冊藍圖（Blueprints）
app.register_blueprint(index_bp)
app.register_blueprint(upload_bp)
app.register_blueprint(video_feed_bp)
app.register_blueprint(video_feed_from_client_bp)

# 顯示 url router 相關資訊
print(f"app.url_map: {app.url_map}")

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
