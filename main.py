import random
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from tensorflow.keras.models import load_model
from fastapi.responses import FileResponse
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Emotion Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "emotion_model.keras"
FRONTEND_PATH = BASE_DIR / "frontend" / "emotion_detector.html"


# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────

model = load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Emotion Configuration
# ─────────────────────────────────────────────────────────────────────────────

emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

emoji_dict = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😃",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}


song_dict = {
    "angry": [
        "Bhaag Dhoondh -- Divine",
        "Kaala Jaadu -- Brahmāstra",
        "Sher Aaya Sher -- Gully Boy",
        "Jhoome Jo Pathaan -- Pathaan",
        "Zinda Banda -- Jawan",
        "Raataan Lambiyan (Remix) -- DJ Shadow",
        "Karma Theme -- Animal",
        "Bezubaan -- ABCD",
        "Apna Time Aayega -- Gully Boy",
        "Bholenath -- Emiway Bantai"
    ],

    "disgust": [
        "Besharam Rang -- Pathaan",
        "Pyaar Hota Kayi Baar Hai -- Tu Jhoothi Main Makkaar",
        "Character Dheela 2.0 -- Shehzada",
        "Naach Meri Rani -- Guru Randhawa",
        "Alcoholia -- Vikram Vedha",
        "Chedkhaniyaan -- Heeramandi",
        "Nasha -- Badshah",
        "Kudiyee Ni Teri -- Selfiee",
        "Yentamma -- Kisi Ka Bhai Kisi Ki Jaan",
        "Jee Ni Karda -- Satyameva Jayate 2"
    ],

    "fear": [
        "Rakt Charitra Theme -- RGV",
        "Azaadi -- Gully Boy",
        "Fear Song -- Devara",
        "Darkhast -- Shivaay",
        "Tandav Theme -- Tandav",
        "Laal Ishq -- Ram-Leela",
        "Aigiri Nandini (Rock Version) -- Mahakaal",
        "Shaitan Ka Saala -- Housefull 4",
        "Khwabon Ke Parindey (Haunting Reprise) -- ZNMD",
        "Jee Le Zaraa (Instrumental) -- Talaash"
    ],

    "happy": [
        "What Jhumka? -- Rocky Aur Rani",
        "Dil Se Dil Tak -- Bawaal",
        "Tere Vaaste -- Zara Hatke Zara Bachke",
        "Heeriye -- Arijit Singh",
        "Jab Se Ishq Hua Hai -- Pooja Verma Studio",
        "Tune Gale Lagaya -- Pooja Verma Studio",
        "Vaaste -- Dhvani Bhanushali",
        "Dance Ka Bhoot -- Brahmāstra",
        "Sauda Khara Khara -- Good Newwz",
        "Gallan Goodiyan -- Dil Dhadakne Do"
    ],

    "neutral": [
        "Qaafirana -- Kedarnath",
        "Kalank Title Track -- Kalank",
        "Jaan Ban Gaye -- Khuda Haafiz",
        "Jitni Dafa -- Parmanu",
        "Nazm Nazm -- Bareilly Ki Barfi",
        "Zaalima -- Raees",
        "Dil Maang Raha Hai -- Ghost",
        "Dariya -- Baar Baar Dekho",
        "Naino Ne Baandhi -- Gold",
        "Aaj Se Teri -- Padman"
    ],

    "sad": [
        "O Bedardeya -- Tu Jhoothi Main Makkaar",
        "Phir Na Milenge -- Broken",
        "Hamari Adhuri Kahani -- Title Track",
        "Toot Gaya -- Broken",
        "Dil Ka Kya -- Metro In Dino",
        "Agar Tum Saath Ho -- Tamasha",
        "Baaton Ko Teri -- All Is Well",
        "Sunn Raha Hai -- Aashiqui 2",
        "Bekhayali -- Kabir Singh",
        "Main Dhoondne Ko Zamaane Mein -- Heartless"
    ],

    "surprise": [
        "Chand Baaliyan -- Aditya",
        "Doobey -- Gehraiyaan",
        "Apna Bana Le -- Bhediya",
        "Naatu Naatu -- RRR (Hindi)",
        "Kesariya (Dance Mix) -- Brahmāstra",
        "Gann Deva -- Street Dancer 3D",
        "Deva Shree Ganesha -- Agneepath",
        "Sadda Dil Vi Tu -- ABCD",
        "Mourya Re -- Don",
        "Shree Ganeshay Dheemahi -- Viruddh"
    ]
}


# ─────────────────────────────────────────────────────────────────────────────
# Face Detector
# ─────────────────────────────────────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction History
# ─────────────────────────────────────────────────────────────────────────────

prediction_history = []
emotion_history = []


# ─────────────────────────────────────────────────────────────────────────────
# Emotion Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_emotion(frame):
    """
    Detect a face in the supplied frame and predict its emotion.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        return {
            "face_detected": False,
            "emotion": "neutral",
            "emoji": "😐",
            "confidence": 0,
            "songs": [],
            "top_emotions": []
        }

    # Select the first detected face
    x, y, w, h = faces[0]

    # Extract face
    face = gray[y:y + h, x:x + w]

    # Resize to model input size
    face = cv2.resize(face, (48, 48))

    # Normalize
    face = face.astype("float32") / 255.0

    # Shape: (1, 48, 48, 1)
    face = np.expand_dims(face, axis=0)
    face = np.expand_dims(face, axis=-1)

    # Model prediction
    prediction = model.predict(face, verbose=0)[0]

    # Temporal smoothing
    prediction_history.append(prediction)

    if len(prediction_history) > 5:
        prediction_history.pop(0)

    prediction = np.mean(prediction_history, axis=0)

    # Existing emotion boost logic
    boost = np.array([
        1.0,   # angry
        2.5,   # disgust
        2.0,   # fear
        1.0,   # happy
        1.0,   # neutral
        1.5,   # sad
        1.8    # surprise
    ])

    prediction = prediction * boost

    prediction_sum = prediction.sum()

    if prediction_sum > 0:
        prediction = prediction / prediction_sum

    # Top 3 emotions
    top_indices = np.argsort(prediction)[::-1][:3]

    top_emotions = [
        {
            "emotion": emotion_labels[i],
            "confidence": round(float(prediction[i] * 100), 2)
        }
        for i in top_indices
    ]

    # Confidence
    confidence = float(np.max(prediction) * 100)

    # Emotion decision
    if confidence < 40:
        emotion = "neutral"
    else:
        current_emotion = emotion_labels[np.argmax(prediction)]

        emotion_history.append(current_emotion)

        if len(emotion_history) > 5:
            emotion_history.pop(0)

        emotion = max(
            set(emotion_history),
            key=emotion_history.count
        )

    # Random songs
    songs = random.sample(
        song_dict[emotion],
        min(3, len(song_dict[emotion]))
    )

    return {
        "face_detected": True,
        "emotion": emotion,
        "emoji": emoji_dict[emotion],
        "confidence": round(confidence, 2),
        "songs": songs,
        "top_emotions": top_emotions,
        "face_box": {
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h)
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

# @app.get("/")
# def root():
#     return {
#         "message": "Emotion Detection API is running"
#     }
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "emotion_detector.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receive an image from the browser and predict the emotion.
    """

    try:
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty image received"
            )

        # Convert bytes → NumPy array
        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        # Decode image
        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image received"
            )

        result = predict_emotion(frame)

        return JSONResponse(result)

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/songs/{emotion}")
def get_songs_by_emotion(emotion: str):

    if emotion not in song_dict:
        raise HTTPException(
            status_code=404,
            detail=f"Emotion '{emotion}' not found. "
                   f"Valid emotions: {emotion_labels}"
        )

    songs = random.sample(
        song_dict[emotion],
        min(3, len(song_dict[emotion]))
    )

    return {
        "emotion": emotion,
        "emoji": emoji_dict[emotion],
        "songs": songs
    }


@app.get("/frontend")
def serve_frontend():

    return FileResponse(
        FRONTEND_PATH
    )


# ─────────────────────────────────────────────────────────────────────────────
# Local Development
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )