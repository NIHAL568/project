from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import csv
import time
from datetime import datetime

app = Flask(__name__)

# Load the trained emotion detection model
model = tf.keras.models.load_model("model.h5")

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Define output folder
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def clear_output_folder():
    """Deletes all files in the output folder."""
    for file in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# Emotion labels (adjust based on your model's classes)
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def detect_emotion(image):
    """Detects emotion from a cropped face using the model."""
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(faces) == 0:
        return None
    
    x, y, w, h = faces[0]  # Consider only the first detected face
    face_img = img_gray[y:y+h, x:x+w]
    face_img = cv2.resize(face_img, (48, 48))  # Adjust size based on model input
    face_img = np.expand_dims(face_img, axis=-1)  # Add channel dimension
    face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
    face_img = face_img / 255.0  # Normalize pixel values
    
    prediction = model.predict(face_img)
    emotion = EMOTION_LABELS[np.argmax(prediction)]
    
    return emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    """Captures images for 10 seconds, determines the average emotion, and saves only the final result."""
    clear_output_folder()  # Clear previous data
    start_time = time.time()
    emotions_detected = []
    
    while time.time() - start_time < 10:
        ret, frame = camera.read()
        if ret:
            emotion = detect_emotion(frame)
            if emotion:
                emotions_detected.append(emotion)
        time.sleep(1)  # Capture an image every second
    
    if not emotions_detected:
        final_emotion = "No face detected"
    else:
        final_emotion = max(set(emotions_detected), key=emotions_detected.count)  # Most frequent emotion
    
    # Save the final captured image
    image_path = os.path.join(OUTPUT_FOLDER, "captured_image.jpg")
    cv2.imwrite(image_path, frame)
    
    # Save final emotion in CSV
    csv_file = os.path.join(OUTPUT_FOLDER, "emotion_results.csv")
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Detected Emotion"])
        writer.writerow([datetime.now(), final_emotion])
    
    return jsonify({"emotion": final_emotion})

@app.route('/video_feed')
def video_feed():
    """Streams the video feed from the camera."""
    def generate():
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    camera = cv2.VideoCapture(0)
    app.run(debug=True)
