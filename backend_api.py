import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"
DEFAULT_THRESHOLD = 0.65
DEFAULT_SMOOTH_WINDOW = 6

app = FastAPI(title="Sign Language Local Backend", version="1.0.0")

state_lock = threading.Lock()
clf = None
landmarker = None
class_labels: List[str] = []
prob_history = deque(maxlen=DEFAULT_SMOOTH_WINDOW)


class PredictResponse(BaseModel):
    hand_detected: bool
    label: str
    raw_label: Optional[str] = None
    confidence: float
    probabilities: Dict[str, float]
    handedness: Optional[str] = None
    handedness_score: Optional[float] = None
    landmarks: Optional[List[Dict[str, float]]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    labels: List[str]


def ensure_model_file(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    print(f"Downloading model to: {model_path}")
    urlretrieve(MODEL_URL, model_path)


def normalize_landmarks(hand_landmarks) -> List[float]:
    wrist = hand_landmarks[0]

    centered = []
    for lm in hand_landmarks:
        centered.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    max_abs = max(abs(v) for v in centered)
    if max_abs == 0:
        return centered

    return [v / max_abs for v in centered]


def decode_image_bytes(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes.")
    return frame


@app.on_event("startup")
def startup_event() -> None:
    global clf, landmarker, class_labels

    if not CLASSIFIER_PATH.exists():
        raise RuntimeError(f"Trained classifier not found: {CLASSIFIER_PATH}")

    ensure_model_file(LANDMARKER_PATH)

    payload = joblib.load(CLASSIFIER_PATH)
    clf = payload["model"]
    class_labels = [str(x) for x in clf.classes_]

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=clf is not None and landmarker is not None,
        labels=class_labels,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    smooth_window: int = Form(DEFAULT_SMOOTH_WINDOW),
    include_landmarks: bool = Form(False),
) -> PredictResponse:
    global prob_history

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (image/* content type).")

    if smooth_window < 1:
        raise HTTPException(status_code=400, detail="smooth_window must be >= 1")

    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        bgr = decode_image_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with state_lock:
        if prob_history.maxlen != smooth_window:
            old = list(prob_history)
            prob_history = deque(old[-smooth_window:], maxlen=smooth_window)

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            return PredictResponse(
                hand_detected=False,
                label="NO_HAND",
                confidence=0.0,
                probabilities={label: 0.0 for label in class_labels},
            )

        hand_landmarks = result.hand_landmarks[0]
        features = normalize_landmarks(hand_landmarks)
        probs = clf.predict_proba(np.array([features], dtype=np.float32))[0]

        prob_history.append(probs)
        avg_probs = np.mean(np.stack(prob_history, axis=0), axis=0)

        pred_idx = int(np.argmax(avg_probs))
        raw_label = str(class_labels[pred_idx])
        pred_conf = float(avg_probs[pred_idx])
        label = raw_label if pred_conf >= threshold else "UNKNOWN"

        probabilities = {str(class_labels[i]): float(avg_probs[i]) for i in range(len(class_labels))}

        handedness = None
        handedness_score = None
        if result.handedness and result.handedness[0]:
            handedness = result.handedness[0][0].category_name
            handedness_score = float(result.handedness[0][0].score)

        landmarks = None
        if include_landmarks:
            landmarks = [
                {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)}
                for lm in hand_landmarks
            ]

    return PredictResponse(
        hand_detected=True,
        label=label,
        raw_label=raw_label,
        confidence=pred_conf,
        probabilities=probabilities,
        handedness=handedness,
        handedness_score=handedness_score,
        landmarks=landmarks,
    )
