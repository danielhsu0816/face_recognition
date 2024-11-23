from flask import Blueprint, request, jsonify, render_template
import cv2
import numpy as np
import base64
from utils.face_recognition_model_utils import detect_faces_and_recognize
from utils.emotion_model_utils import predict_emotion_from_frame

video_feed_from_client_bp = Blueprint('video_feed_from_client', __name__)

@video_feed_from_client_bp.route('/video_feed_from_client')
def video_feed_page():
    """
    渲染網頁，提供客戶端影像捕捉功能。
    """
    return render_template('video_feed_from_client.html')

@video_feed_from_client_bp.route('/process_image', methods=['POST'])
def process_image():
    """
    接收客戶端的 Base64 影像數據，並返回二值化與情緒與人像偵測影像的處理結果。
    """
    try:
        # 獲取 Base64 影像數據
        data = request.json.get('image')
        if not data:
            return jsonify({"error": "No image data received"}), 400

        # 解碼 Base64 成 OpenCV 可處理的影像
        image_data = base64.b64decode(data.split(',')[1])
        np_array = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # 轉為灰階影像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 生成二值化影像
        binary_output = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # 偵測人臉並進行情緒辨識
        faces = detect_faces_and_recognize(frame)
        for (x, y, w, h, text) in faces:
            # 畫出人臉框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # 提取人臉區域並進行情緒辨識
            face_gray = gray[y:y + h, x:x + w]
            emotion, confidence = predict_emotion_from_frame(face_gray)

            # 在影像上顯示人名和情緒
            cv2.putText(
                frame, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
            )
            cv2.putText(
                frame, f"{emotion}, Conf.{confidence:.1f}%", (x, y + h + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        # 將處理過的影像編碼為 Base64 格式
        _, frame_encoded = cv2.imencode('.jpg', frame)
        _, binary_encoded = cv2.imencode('.jpg', binary_output)
        frame_base64 = base64.b64encode(frame_encoded).decode('utf-8')
        binary_base64 = base64.b64encode(binary_encoded).decode('utf-8')

        # 返回處理結果
        return jsonify({
            "original_image": f"data:image/jpeg;base64,{frame_base64}",
            "binary_image": f"data:image/jpeg;base64,{binary_base64}"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
