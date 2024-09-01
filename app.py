from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import webbrowser
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
app = Flask(__name__)

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

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        finger_count = detect_fingers(frame)
        if finger_count == 1:
            webbrowser.open('https://www.youtube.com')
        elif finger_count == 2:
            webbrowser.open('https://www.google.com')

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
