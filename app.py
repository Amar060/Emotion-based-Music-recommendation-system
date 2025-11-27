from flask import Flask, render_template, Response, jsonify
import cv2
import sys # Import sys for better error handling
# Note: Ensure emotion.py and music.py are in your project folder
from emotion import ReliableEmotionDetector
from music import recommend_music

# --- Initialization and Error Checks ---
# The camera should be initialized once globally
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("FATAL ERROR: Could not open the camera. Check if it's connected, the correct index (0) is used, or if another app is using it.", file=sys.stderr)
    # The application will likely crash later if the camera isn't ready.

try:
    # IMPORTANT: Ensure 'final_emotion_model.h5' is in your project root
    detector = ReliableEmotionDetector('final_emotion_model.h5')
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize emotion detector. Check model path and file integrity: {e}", file=sys.stderr)
    # The application will likely crash later if the model isn't ready.

app = Flask(__name__)

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    # This loop runs constantly to stream video to the HTML <img src="/video"> tag
    while True:
        success, frame = camera.read()
        if not success:
            # If the stream fails, try to re-open the camera or break
            # In production, you might try re-initializing the camera here
            break

        # Detect emotion and draw box/text
        emotion, confidence, box = detector.detect_emotion(frame)
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if emotion:
                # Format the confidence percentage display
                text = f"{emotion} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Encode the frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    # Route for the live video stream (runs generate_frames)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recommend')
def recommend():
    # Route triggered by the button click
    success, frame = camera.read()
    
    # --- CRITICAL ERROR CHECK ---
    # The camera might be open, but still fail to read a frame (especially on first try)
    if not success or frame is None:
        print("ERROR: Failed to read frame for recommendation. Camera might be busy.", file=sys.stderr)
        return jsonify({
            'emotion': 'Camera Error', 
            'song': 'Failed to capture frame. Please check the camera and restart the server.'
        })
        
    try:
        emotion, _, _ = detector.detect_emotion(frame)
        
        if emotion:
            song = recommend_music(emotion)
            # Return a JSON object for the JavaScript to easily parse
            return jsonify({'emotion': emotion, 'song': song})
            
        # If no face is detected
        return jsonify({'emotion': 'No Face Detected', 'song': 'Please ensure your face is clearly visible in the frame.'})

    except Exception as e:
        # Catch any other runtime error during detection/recommendation
        print(f"RUNTIME ERROR in /recommend route: {e}", file=sys.stderr)
        return jsonify({
            'emotion': 'App Crash',
            'song': 'An internal error occurred. Check the VS Code terminal for details.'
        }), 500

if __name__ == "__main__":
    app.run(debug=True)