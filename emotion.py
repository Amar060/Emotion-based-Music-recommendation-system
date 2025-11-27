from tensorflow.keras.models import load_model
import cv2
import numpy as np

class ReliableEmotionDetector:
    def __init__(self, model_path='final_emotion_model.h5'):
        self.model = load_model(model_path)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_emotion(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None, None, None

        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=(0, -1))

        preds = self.model.predict(roi, verbose=0)
        emotion = self.emotions[np.argmax(preds)]
        confidence = np.max(preds)
        return emotion, confidence, (x, y, w, h)
