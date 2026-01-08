# Real-Time Face Analysis & Recognition (OpenCV + DeepFace)

This project is a real-time face analysis and face recognition system using a webcam.
It detects faces, analyzes age group, gender, and emotion, and performs face recognition with persistent memory using DeepFace embeddings.

Unknown faces can be saved dynamically during runtime and will be remembered across sessions.

---

## Features

- Live webcam face detection using OpenCV Haar Cascade
- Face analysis using DeepFace
  - Age estimation (grouped)
  - Gender detection
  - Emotion recognition
- Face recognition using FaceNet embeddings
- Persistent storage of known faces using pickle
- Pause and resume video stream
- Add new faces dynamically without manual input
- Console logging for every observation

---

## Technologies Used

- Python 3
- OpenCV (cv2)
- DeepFace
- NumPy
- TensorFlow
- Haar Cascade Classifier

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/aarx09/face_recog_pyhton.git
cd face_recog_pyhton
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install opencv-python deepface tensorflow numpy
```

---

## Usage

Run the script:
```bash
python face_analysis.py
```

A window titled "Face Analysis" will open and start using your webcam.

---

## Controls

| Key | Action |
|-----|--------|
| q   | Quit the application |
| ESC | Quit the application |
| p   | Pause or resume video |
| r   | Save the last detected unknown face |

---

## Stored Data

Known faces are saved in:
```text
known_faces.pkl
```

This file allows face data to persist across sessions.

---

## Notes and Limitations

- Lighting conditions affect accuracy
- Emotion detection is probabilistic
- Haar Cascade may miss faces at extreme angles
- Not intended for security-critical applications

---

## License

This project is intended for educational and experimental purposes.
