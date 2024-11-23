from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2 as cv

# 全域變數
trained_model = None
input_details = None
output_details = None
Emotion_Rule = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

def initialize_emotion_model(app):
    """
    初始化情感辨識模型，從 Flask 應用配置中讀取 TensorFlow Lite 模型路徑。
    """
    global interpreter, input_details, output_details
    model_path = app.config['MODEL_PATH']
    
    # 使用 TFLite 解釋器載入模型
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # 獲取模型的輸入和輸出資訊
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

def read_plot_and_test(image_path):
    """
    使用 TensorFlow Lite 模型對圖片進行情感分析。
    :param image_path: 圖片的檔案路徑
    :return: 預測的情感標籤和置信度百分比
    """
    global interpreter, input_details, output_details, Emotion_Rule
    if interpreter is None:
        raise RuntimeError("The model is not initialized. Call initialize_emotion_model() first.")

    # 讀取影像並進行處理
    test_img_bgr = cv.imread(image_path)
    test_img_rbg = cv.cvtColor(test_img_bgr, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img_rbg, cv.COLOR_RGB2GRAY)
    test_img_resize = cv.resize(test_img_gray, (48, 48), interpolation=cv.INTER_NEAREST)
    test_sample = test_img_resize.reshape(1, 48, 48, 1).astype(np.float32) / 255.0

    # 設置輸入數據並進行推論
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()

    # 獲取輸出數據
    prediction = interpreter.get_tensor(output_details[0]['index']).reshape(-1)
    index = np.argmax(prediction)
    conf = prediction[index]

    return Emotion_Rule[index], round(conf * 100, 2)

def predict_emotion_from_frame(face_gray):
    """
    使用 TensorFlow Lite 模型對灰階人臉進行情感辨識。
    :param face_gray: 灰階人臉圖像 (需為 48x48)
    :return: 預測的情感標籤和信賴度百分比
    """
    global interpreter, input_details, output_details, Emotion_Rule
    if interpreter is None:
        raise RuntimeError("The model is not initialized. Call initialize_emotion_model() first.")

    # 將灰階人臉調整為模型輸入格式
    face_resized = cv.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_input = face_normalized.reshape(1, 48, 48, 1).astype(np.float32)

    # 設置輸入數據並進行推論
    interpreter.set_tensor(input_details[0]['index'], face_input)
    interpreter.invoke()

    # 獲取輸出數據
    prediction = interpreter.get_tensor(output_details[0]['index']).reshape(-1)
    index = np.argmax(prediction)
    emotion = Emotion_Rule[index]
    confidence = round(prediction[index] * 100, 2)

    return emotion, confidence