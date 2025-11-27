# Emotion-Based Music Recommendation System 🎧😄

This project detects a user's **real-time facial emotion** using a webcam and recommends music that best suits their mood.  
It combines **Deep Learning + Computer Vision + Web App Deployment** in one complete application.

---

## ✨ Features
- 🎥 Live webcam feed processing
- 😁 Emotion detection into 7 classes:  
  *Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral*
- 🎵 Smart mood-based song suggestions
- ⚡ Real-time performance using Flask + OpenCV
- 🤖 Trained CNN model included (`final_emotion_model.h5`)

---

## 🧠 Technology Stack

| Area | Technology |
|------|------------|
| Deep Learning | TensorFlow, Keras |
| Computer Vision | OpenCV |
| Web Framework | Flask |
| Programming | Python |
| Dataset | FER-2013 (48x48 grayscale faces) |

---

## 📌 System Architecture

User → Webcam → Face Detection → Emotion Prediction → Music Recommendation → UI Display (Web App)

---


- `emotion.py` → Haar Cascade + CNN inference  
- `music.py` → Playlist recommendation logic  
- `app.py` → Flask backend + video streaming  
- `train.py` → CNN training script

---

## Project Structure

Emotion-based-Music-recommendation-system/
→ app.py                 # Flask web server (runs the app + video feed + API)
→ emotion.py             # Emotion detection class (Haar Cascade + CNN prediction)
→ music.py               # Dictionary-based song recommendation logic
→ train.py               # CNN training script for FER-2013 dataset
→ final_emotion_model.h5 # Trained emotion recognition model
→ templates/             # Frontend HTML UI
  → index.html           # Main webpage displaying live camera stream
→ README.md              # Documentation

---

## ▶️ How to Run

### 1️⃣ Install Libraries
  pip install flask opencv-python numpy tensorflow

### 2️⃣ Start The Web App
  python app.py

### 3️⃣ Open in Browser
  http://127.0.0.1:5000/

---

## Song Recommendation Logic

Each emotion maps to songs. Example:

| Emotion | Suggested Song |
|--------|----------------|
| Happy | Happy – Pharrell Williams |
| Sad | Fix You – Coldplay |
| Angry | In The End – Linkin Park |
| Neutral | Perfect – Ed Sheeran |

(Full song dictionary is in music.py)

---

## Model Details

- Trained on FER-2013 facial emotion dataset
- CNN architecture:
  - 3 Conv2D blocks (64, 128, 256 filters)
  - BatchNormalization + MaxPooling + Dropout
  - Dense(512) classifier with Dropout
  - Softmax output for 7 classes
- Achieved ~60% validation accuracy
- Model saved as: final_emotion_model.h5

---

## Future Work

- Integrate Spotify / YouTube Data API
- Use MTCNN / YOLO for face detection
- Improve model accuracy with ResNet/MobileNet
- Personalized playlist generation

---

📝 License:

This project is intended for educational and research purposes.
💡 If you like this project, don’t forget to ⭐ the repo!
