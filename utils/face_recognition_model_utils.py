import cv2

# 全域變數
face_cascade = None
recognizer = None

# ID 對應人名的字典
name = {
    '1': 'Harris',  # ID 1 對應 Harris
    '2': 'Trump',   # ID 2 對應 Trump
    '3': 'Daniel'   # ID 3 對應 Daniel
}

def initialize_faceRecognition_resources(app):
    """
    初始化全域變數 face_cascade 和 recognizer，包含人臉偵測器和辨識模型。
    透過 Flask app 的配置來載入模型。
    """
    global face_cascade, recognizer
    cascade_path = app.config['CASCADE_PATH']
    face_model_path = app.config['FACE_MODEL_PATH']
    
    # 載入人臉檢測模型
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # 載入人臉辨識模型
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(face_model_path) 

def detect_faces_and_recognize(img):
    """
    偵測人臉並進行人名辨識。
    :param img: 原始影像
    :return: 偵測到的人臉框與對應人名的列表 [(x, y, w, h, name), ...]
    """
    global face_cascade, recognizer
    if face_cascade is None or recognizer is None:
        raise RuntimeError("Resources not initialized. Call initialize_resources() first.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    results = []
    for (x, y, w, h) in faces_detected:
        # 使用訓練好的模型進行辨識
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # 映射 ID 到人名
        person_name = name.get(str(idnum), "Unknown")
        if confidence < 70:  # 信心值低於 70 時，設為未知
            person_name = "Who are You?"
        text = f"{person_name}, Conf.{confidence:.1f}%" 
        results.append((x, y, w, h, text))
    return results