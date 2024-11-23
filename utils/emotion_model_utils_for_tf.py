import tensorflow as tf 
import numpy as np
import cv2 as cv

# 全域變數
trained_model = None
Emotion_Rule = {
    0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy",
    4: "Sad", 5: "Surprise", 6: "Neutral"
}

def initialize_emotion_model(app):
    """
    初始化情感辨識模型，從 Flask 應用配置中獲取模型路徑。 
    
    p.s. for import tensorflow as tf 版本
    """
    global trained_model
    model_path = app.config['MODEL_PATH']
    trained_model = tf.keras.models.load_model(model_path)

def read_plot_and_test(image_path):
    """
    使用訓練好的模型對圖片進行情感分析。

    p.s. for import tensorflow as tf 版本
    """
    global trained_model
    if trained_model is None:
        raise RuntimeError("The model is not initialized. Call initialize_model() first.")

    # 讀取影像並進行處理
    test_img_bgr = cv.imread(image_path) 
    test_img_rbg = cv.cvtColor(test_img_bgr, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img_rbg, cv.COLOR_RGB2GRAY)
    test_img_resize = cv.resize(test_img_gray, (48, 48), interpolation=cv.INTER_NEAREST)
    test_sample = test_img_resize.reshape(1, 48, 48, 1) / 255.0

    # 模型預測
    prediction = trained_model.predict(test_sample).reshape(-1)
    index = np.argmax(prediction)
    conf = prediction[index]
    return Emotion_Rule[index], round(conf * 100, 2)

def predict_emotion_from_frame(face_gray):
    """
    使用情感模型對灰階人臉進行情感辨識。
    :param face_gray: 灰階人臉圖像 (需為 48x48)
    :return: 預測的情感標籤和信賴度百分比 
    
    p.s. for import tensorflow as tf 版本
    """
    global trained_model, Emotion_Rule
    if trained_model is None:
        raise RuntimeError("The model is not initialized. Call initialize_model() first.")

    # 將灰階人臉調整為模型輸入格式
    face_resized = cv.resize(face_gray, (48, 48))
    face_normalized = face_resized / 255.0
    face_input = face_normalized.reshape(1, 48, 48, 1)

    # 使用模型進行預測
    prediction = trained_model.predict(face_input).reshape(-1)
    index = np.argmax(prediction)
    emotion = Emotion_Rule[index]
    confidence = round(prediction[index] * 100, 2)

    return emotion, confidence