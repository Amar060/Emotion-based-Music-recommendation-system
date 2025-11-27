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

## 📂 Project Structure

```bash
Emotion-based-Music-recommendation-system/
├─ app.py                 # Flask web server
├─ emotion.py             # Emotion detector (face + CNN)
├─ music.py               # Mood-based song mapping
├─ train.py               # CNN training script
├─ final_emotion_model.h5 # Trained CNN model file (optional in repo)
├─ templates/
│  └─ index.html          # Frontend page
└─ README.md
---

## ▶️ How to Run

### 1️⃣ Install Libraries
pip install flask opencv-python numpy tensorflow

### 2️⃣ Start The Web App
python app.py

### 3️⃣ Open in Browser
http://127.0.0.1:5000/

---

🎵 Song Recommendation Logic

Each emotion maps to multiple predefined songs.

Emotion	Example Suggested Song
Happy	Happy – Pharrell Williams
Sad	Fix You – Coldplay
Angry	In The End – Linkin Park
Neutral	Perfect – Ed Sheeran
Surprise	Adventure of a Lifetime – Coldplay

(Full mapping is defined in music.py)

---

📊 Model Information

Trained on FER-2013 Dataset
Achieved ~60%+ validation accuracy

CNN layers:
  -3× Conv2D blocks (64 → 128 → 256 filters)
  -BatchNorm + MaxPooling + Dropout
  -Dense(512) + Dropout
  -Output: Softmax(7)

Model saved as: final_emotion_model.h5

---

🚀 Future Enhancements: 

> Spotify or YouTube API integration
> Better deep-learning based face detector (MTCNN / YOLO)
> More accurate model architecture (ResNet / MobileNet)
> Personalized dynamic playlists

---

📝 License:

This project is intended for educational and research purposes.
💡 If you like this project, don’t forget to ⭐ the repo!
