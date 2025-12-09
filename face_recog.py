import os
import cv2
import pickle
from deepface import DeepFace
import numpy as np
from datetime import datetime

# ----------------------------
# Environment setup
# ----------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # suppress TF info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN optimizations for consistency

KNOWN_FACES_FILE = 'known_faces.pkl'

# ----------------------------
# Persistence: load known faces
# ----------------------------
if os.path.exists(KNOWN_FACES_FILE):
    with open(KNOWN_FACES_FILE, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# ----------------------------
# Utility functions
# ----------------------------
def get_age_class(age):
    if age < 10: return '0-9'
    elif age < 20: return '10-19'
    elif age < 30: return '20-29'
    elif age < 40: return '30-39'
    elif age < 50: return '40-49'
    elif age < 60: return '50-59'
    else: return '60+'

def remember_face(face_encoding, name):
    known_face_encodings.append(face_encoding)
    known_face_names.append(name)
    with open(KNOWN_FACES_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def compute_distance(a, b):
    return np.linalg.norm(a - b)

def best_match(face_encoding, encodings, threshold=0.6):
    if not encodings:
        return None, None
    dists = [compute_distance(e, face_encoding) for e in encodings]
    idx = int(np.argmin(dists))
    if dists[idx] < threshold:
        return idx, dists[idx]
    return None, None

# ----------------------------
# Video capture setup
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

paused = False
last_unknown_face_encoding = None

print("Controls: [q] quit | [ESC] quit | [p] pause/resume | [r] remember last UNKNOWN face")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("Frame grab failed. Exiting.")
            break

        # Brightness/contrast tuning
        alpha, beta = 1.3, 35
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        # ----------------------------
        # Face attribute analysis
        # ----------------------------
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )
        except Exception:
            analysis = None

        if analysis:
            result = analysis[0] if isinstance(analysis, list) else analysis

            age_class = get_age_class(result['age'])
            gender = result['dominant_gender']   # ✅ dominant gender
            emotion = result['dominant_emotion']

            if emotion == 'neutral' and result['emotion'].get('neutral', 0) < 0.8:
                emotion = max(result['emotion'], key=result['emotion'].get)

            # Overlay attributes on frame
            cv2.putText(frame, f'Age: {age_class}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f'Gender: {gender}', (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f'Emotion: {emotion}', (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 👉 Print to console after every observation
            print(f"Observation -> Age: {age_class}, Gender: {gender}, Emotion: {emotion}")

        # ----------------------------
        # Face detection + recognition
        # ----------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        last_unknown_face_encoding = None

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]

            try:
                rep = DeepFace.represent(
                    roi,
                    model_name='Facenet',
                    detector_backend='skip'
                )
                embedding = np.array(rep['embedding'])
            except Exception:
                continue

            idx, dist = best_match(embedding, known_face_encodings, threshold=0.6)
            if idx is not None:
                name = known_face_names[idx]
                color = (0, 200, 255)
                label = f"{name}"
                print(f"Recognized: {name}")
            else:
                name = "Unknown"
                color = (0, 255, 0)
                label = "Unknown"
                last_unknown_face_encoding = embedding
                print("Recognized: Unknown")

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if last_unknown_face_encoding is not None:
            cv2.putText(frame, 'Press [r] to remember this face',
                        (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ✅ Live window showing recognition
        cv2.imshow('Face Analysis', frame)

    # ----------------------------
    # Key handling
    # ----------------------------
    key = cv2.waitKey(50) & 0xFF  # longer wait for reliable key capture
    if key == ord('q') or key == 27:  # q or ESC
        print("Quitting...")
        break
    elif key == ord('p'):  # pause/resume
        paused = not paused
        print(f"Paused: {paused}")
    elif key == ord('r'):  # remember face (auto-name, no console input)
        if last_unknown_face_encoding is not None:
            name = f"Person_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            remember_face(last_unknown_face_encoding, name)
            print(f"Saved new identity: {name}")

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
