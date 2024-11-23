from flask import Blueprint, Response, render_template
from utils.face_recognition_model_utils import detect_faces_and_recognize
from utils.emotion_model_utils import predict_emotion_from_frame
import cv2
import threading

video_feed_bp = Blueprint('video_feed', __name__)

# 全域攝影機變數與執行緒
camera = None
frame = None
lock = threading.Lock()

def init_camera():
    """
    初始化攝像頭。
    """
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("無法打開攝像頭，請檢查設備或驅動程式。")
    
def release_camera():
    """
    釋放攝像頭資源。
    """
    global camera
    if camera is not None and camera.isOpened():
        camera.release()    

def capture_frames():
    """
    持續捕獲攝像頭影像。
    """
    global frame, camera
    while camera and camera.isOpened():
        with lock:
            ret, frame = camera.read()
            if not ret:
                frame = None

# 啟動攝影機擷取線程   
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()             

@video_feed_bp.route('/video_feed/original')
def video_feed_original():
    """
    路由：即時影像辨識結果（人臉與情緒框）。
    """
    init_camera()
    thread = threading.Thread(target=capture_frames, daemon=True)
    thread.start()
    return Response(generate_original_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_feed_bp.route('/video_feed/binary')
def video_feed_binary():
    """
    路由：二值化影像處理結果。
    """
    init_camera()
    thread = threading.Thread(target=capture_frames, daemon=True)
    thread.start()
    return Response(generate_binary_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@video_feed_bp.route('/video_feed')
def video_feed_page():
    """
    渲染網頁，同時展示兩個即時影像。
    """
    return render_template('video_feed.html')

def generate_original_frames():
    """
    生成即時影像辨識結果，包括人臉與情緒辨識。
    """
    global frame
    try:
        while True:
            with lock:
                if frame is None:
                    continue
                img = frame.copy()

            # 轉為灰階影像，方便人臉與情緒辨識
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 偵測人臉
            results = detect_faces_and_recognize(img)  # 偵測到的結果列表 (x, y, w, h, id)
            for (x, y, w, h, text) in results:
                # 畫出人臉框
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 提取人臉區域並進行情緒辨識
                face_gray = gray[y:y+h, x:x+w]
                emotion, confidence = predict_emotion_from_frame(face_gray)

                # 顯示人名
                cv2.putText(
                    img, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
                )

                # 顯示情緒
                emotion_text = f"Emo.{emotion}, Conf.{confidence:.1f}%"
                cv2.putText(
                    img, emotion_text, (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
                )

            # 將影像編碼為 JPEG 格式並生成流數據
            _, buffer = cv2.imencode('.jpg', img)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    finally:
        release_camera()    

def generate_binary_frames():
    """
    生成二值化影像。
    """
    global frame
    try:
        while True:
            with lock:
                if frame is None:
                    continue
                img = frame.copy()

            # 轉為灰階影像並進行二值化處理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            binary_output = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            _, buffer = cv2.imencode('.jpg', binary_output)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    finally:
        release_camera()
