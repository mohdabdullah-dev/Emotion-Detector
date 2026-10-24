# Emotion Detection System — FastAPI

A real-time facial emotion detection system built with a CNN model, wrapped in a FastAPI backend. Detects emotions via webcam and recommends Hindi songs based on your mood.

---

## Features

- Live webcam feed with face detection
- CNN-based emotion recognition (7 emotions)
- Song recommendations based on detected emotion
- REST API with MJPEG video stream

---

## Emotions Supported

| Emotion  | Emoji |
|----------|-------|
| Happy    | 😃    |
| Sad      | 😢    |
| Angry    | 😠    |
| Fear     | 😨    |
| Surprise | 😲    |
| Disgust  | 🤢    |
| Neutral  | 😐    |

---

## Project Structure

```
emotion-detection-api/
├── main.py               # FastAPI app
├── requirements.txt      # Dependencies
├── models/
│   └── emotion_model.keras   # Your trained CNN model (add manually)
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your model

Place your trained model file inside the `models/` folder:

```
models/emotion_model.keras
```

> The model is excluded from GitHub via `.gitignore` because of file size. Download it separately or use Git LFS.

### 3. Run the server

```bash
uvicorn main:app --reload
```

---

## API Endpoints

| Method | Endpoint              | Description                          |
|--------|-----------------------|--------------------------------------|
| GET    | `/`                   | Health check                         |
| GET    | `/emotion`            | Current emotion + emoji + songs      |
| GET    | `/stream`             | Live MJPEG webcam stream             |
| GET    | `/songs/{emotion}`    | Songs for a specific emotion         |

### Example Response — `/emotion`

```json
{
  "emotion": "happy",
  "emoji": "😃",
  "songs": [
    "What Jhumka? -- Rocky Aur Rani",
    "Vaaste -- Dhvani Bhanushali",
    "Dance Ka Bhoot -- Brahmāstra"
  ]
}
```

---

## Tech Stack

- Python 3.10+
- FastAPI
- TensorFlow / Keras
- OpenCV
- NumPy

---

## Model Info

The CNN model was trained on the FER-2013 dataset to classify 48x48 grayscale face images into 7 emotion categories.
