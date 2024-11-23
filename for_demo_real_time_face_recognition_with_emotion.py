import cv2
import tensorflow as tf
import numpy as np
import os

# 初始化人臉辨識器，載入訓練好的模型
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('Opencv_Images/face.yml')  # 載入訓練好的模型

# 載入人臉追蹤模型（檢查路徑是否正確）
cascade_path = cv2.data.haarcascades + "Opencv_Images/haarcascade_frontalface_default.xml"
if not os.path.exists(cascade_path):
    print("Haarcascade 檔案未找到，請檢查路徑:", cascade_path)
    exit()

face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    print("CascadeClassifier 加載失敗，請確認 XML 檔案內容或路徑正確。")
    exit()

# 載入情緒辨識模型
trained_model = tf.keras.models.load_model('trained_model.keras')

# 情緒與編號的對照表
Emotion_Rule = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# 建立姓名與 ID 的對照表
name = {
    '1': 'Harris',  # ID 1 對應 Harris
    '2': 'Trump',   # ID 2 對應 Trump
    '3': 'Daniel'   # ID 3 對應 Daniel
}

# 選擇來源：影片檔案或攝影機
video_file = "path_to_your_video.mp4"  # 直接指定影片檔案路徑

cap = cv2.VideoCapture(video_file)  # 嘗試打開影片檔案
if not cap.isOpened():  # 如果影片檔案無法開啟，使用即時攝影機
    print("無法開啟影片檔案，轉為使用即時攝影機")
    cap = cv2.VideoCapture(0)  # 使用即時攝影機
    if not cap.isOpened():  # 如果即時攝影機也無法開啟
        print("無法開啟攝影機")
        exit()

# 即時辨識與情緒分析
while True:
    ret, img = cap.read()
    if not ret:
        print("無法讀取影像")
        break

    # 縮小影像尺寸以提高處理速度
    img = cv2.resize(img, (640, 480))

    # 轉為灰階影像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 自適應二值化影像（可選）
    gray_blurred = cv2.medianBlur(gray, 5)
    binary_output = cv2.adaptiveThreshold(
        gray_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # 偵測人臉
    faces_detected = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # 如果未檢測到人臉，輸出提示
    if len(faces_detected) == 0:
        print("未檢測到人臉")

    # 對每張偵測到的臉進行辨識與情緒分析
    for (x, y, w, h) in faces_detected:
        # 標記人臉外框
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # 使用訓練好的模型進行人臉辨識
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # 檢查可信性並調整閾值
        region = gray[y:y+h, x:x+w]
        print('region:' + region)
        mean_intensity = region.mean()  # 計算區域均值
        if confidence < 70 and str(idnum) in name and w * h > 300 and mean_intensity > 50:
            text = name[str(idnum)]
        else:
            text = 'Who are you?'

        # 在人臉框旁加上名字
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 進行情緒分析
        face_region = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (48, 48), interpolation=cv2.INTER_NEAREST)
        face_resized = face_resized.reshape(1, 48, 48, 1) / 255.0  # 正規化
        
        # 預測情緒
        emotion_prediction = trained_model(face_resized).numpy().reshape(-1)
        emotion_index = np.argmax(emotion_prediction)
        emotion_confidence = emotion_prediction[emotion_index]
        emotion_text = f"{Emotion_Rule[emotion_index]} ({round(emotion_confidence * 100, 2)}%)"

        # 在影像上顯示情緒
        cv2.putText(img, emotion_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # 同時顯示原始影像（含人臉辨識框）和二值化影像
    cv2.imshow('Face & Emotion Recognition', img)
    cv2.imshow('Adaptive Threshold (Binary)', binary_output)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放攝影機並關閉所有視窗
cap.release()
cv2.destroyAllWindows()
