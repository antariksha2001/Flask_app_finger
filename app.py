import os
import cv2
import numpy as np  # Add this line to import numpy
import mediapipe as mp
import webbrowser
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def detect_fingers(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            finger_count = sum([1 for lm in hand_landmarks.landmark[1:5] if lm.y < hand_landmarks.landmark[0].y])
            return finger_count
    return 0

@socketio.on('video_frame')
def handle_video_frame(data):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    finger_count = detect_fingers(frame)
    if finger_count == 1:
        webbrowser.open('https://www.youtube.com')
    elif finger_count == 2:
        webbrowser.open('https://www.google.com')
    _, buffer = cv2.imencode('.jpg', frame)
    emit('response_back', buffer.tobytes())

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port)
