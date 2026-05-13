import threading
from collections import deque, Counter
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve
import time
from contextlib import asynccontextmanager

import cv2
import joblib
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
import calibrator
from accounts import AccountManager

account_manager = AccountManager()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Constants
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"
DEFAULT_THRESHOLD = 0.70
WINDOW_SIZE = 30
FEATURE_COUNT = 63
MOTION_LETTERS = ['J', 'Z']

# Global State
state_lock = threading.Lock()
clf_frame = None
clf_window = None
landmarker = None
class_labels: List[str] = []

# Sliding window for landmarks (Memory)
landmark_history = deque(maxlen=WINDOW_SIZE)

class PredictResponse(BaseModel):
    hand_detected: bool
    label: str
    prediction_type: str  # "Vote", "Motion", or "Buffering"
    confidence: float
    probabilities: Dict[str, float]
    handedness: Optional[str] = None
    landmarks: Optional[List[Dict[str, float]]] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    labels: List[str]

class CalibrateResponse(BaseModel):
    success: bool
    message: str
    hand_detected: bool

class RetrainResponse(BaseModel):
    success: bool
    message: str
    accuracy: float
    calibration_samples_used: int
    labels: List[str]

class AccountCreateRequest(BaseModel):
    username: str
    password: str
    name: str
    mastery_level: str = "Beginner"

class AccountLoginRequest(BaseModel):
    username: str
    password: str

class AccountUpdateProgressRequest(BaseModel):
    username: str
    letter: str
    level: int

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
    if max_abs == 0: return centered
    return [v / max_abs for v in centered]

def decode_image_bytes(file_bytes: bytes):
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes.")
    return frame

@asynccontextmanager
async def lifespan(app: FastAPI):
    global clf_frame, clf_window, landmarker, class_labels
    if not CLASSIFIER_PATH.exists():
        print(f"Warning: Classifier not found at {CLASSIFIER_PATH}. Run training first.")
    else:
        ensure_model_file(LANDMARKER_PATH)
        payload = joblib.load(CLASSIFIER_PATH)
        clf_frame = payload["frame_model"]
        clf_window = payload["window_model"]
        class_labels = [str(x) for x in payload["labels"]]

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        landmarker = HandLandmarker.create_from_options(options)
    
    yield
    # Cleanup can be added here if needed

app = FastAPI(title="Sign Language Motion Backend", version="1.1.0", lifespan=lifespan)

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=clf_window is not None,
        labels=class_labels,
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
    include_landmarks: bool = Form(False),
) -> PredictResponse:
    global landmark_history

    file_bytes = await file.read()
    try:
        bgr = decode_image_bytes(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with state_lock:
        if landmarker is None or clf_window is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            landmark_history.clear() # Clear memory if hand is lost
            return PredictResponse(
                hand_detected=False,
                label="NO_HAND",
                prediction_type="None",
                confidence=0.0,
                probabilities={label: 0.0 for label in class_labels},
            )

        hand_landmarks = result.hand_landmarks[0]
        features = normalize_landmarks(hand_landmarks)
        landmark_history.append(features)

        # Consensus Logic
        label = "ANALYZING"
        pred_type = "Buffering"
        pred_conf = 0.0
        probabilities = {label: 0.0 for label in class_labels}

        if len(landmark_history) == WINDOW_SIZE:
            # 1. Motion Prediction
            window_flat = []
            for f in landmark_history: window_flat.extend(f)
            w_probs = clf_window.predict_proba(np.array([window_flat], dtype=np.float32))[0]
            w_idx = int(np.argmax(w_probs))
            w_label = clf_window.classes_[w_idx]
            w_conf = float(w_probs[w_idx])

            # 2. Voting Prediction
            f_batch = np.array(list(landmark_history), dtype=np.float32)
            f_preds = clf_frame.predict(f_batch)
            vote_counts = Counter(f_preds)
            v_label, v_count = vote_counts.most_common(1)[0]
            v_conf = v_count / WINDOW_SIZE

            # 3. Decision
            if w_label in MOTION_LETTERS and w_conf > 0.60:
                label = w_label
                pred_conf = w_conf
                pred_type = "Motion"
            elif v_conf >= threshold:
                label = v_label
                pred_conf = v_conf
                pred_type = "Vote"
            else:
                label = "UNCERTAIN"
                pred_conf = v_conf
                pred_type = "Mixed"
            
            probabilities = {str(clf_window.classes_[i]): float(w_probs[i]) for i in range(len(clf_window.classes_))}

        # Meta Data
        handedness = result.handedness[0][0].category_name if result.handedness else None
        landmarks = [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand_landmarks] if include_landmarks else None

    return PredictResponse(
        hand_detected=True,
        label=label,
        prediction_type=pred_type,
        confidence=pred_conf,
        probabilities=probabilities,
        handedness=handedness,
        landmarks=landmarks,
    )

@app.post("/calibrate", response_model=CalibrateResponse)
async def calibrate_endpoint(file: UploadFile = File(...), label: str = Form(...)) -> CalibrateResponse:
    # Basic static calibration (stays single-frame for now)
    file_bytes = await file.read()
    bgr = decode_image_bytes(file_bytes)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with state_lock:
        result = landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return CalibrateResponse(success=False, message="No hand detected", hand_detected=False)

        features = normalize_landmarks(result.hand_landmarks[0])
        calibrator.add_calibration_sample(features, label)

    return CalibrateResponse(success=True, message=f"Added '{label}'", hand_detected=True)

@app.post("/retrain", response_model=RetrainResponse)
async def retrain_endpoint() -> RetrainResponse:
    global clf_frame, clf_window, class_labels
    base_csv_path = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"
    
    with state_lock:
        try:
            new_clf, new_labels, accuracy, samples_used = calibrator.retrain_model(base_csv_path, CLASSIFIER_PATH)
            # Re-load the dual models from the newly saved file
            payload = joblib.load(CLASSIFIER_PATH)
            clf_frame = payload["frame_model"]
            clf_window = payload["window_model"]
            class_labels = [str(x) for x in payload["labels"]]
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return RetrainResponse(
        success=True, message="Retrained unified model", accuracy=accuracy,
        calibration_samples_used=samples_used, labels=class_labels
    )

@app.post("/account/create", tags=["Account"])
async def create_account(req: AccountCreateRequest):
    success, msg = account_manager.create_account(req.username, req.password, req.name, req.mastery_level)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}

@app.post("/account/login", tags=["Account"])
async def login_account(req: AccountLoginRequest):
    success, data = account_manager.login(req.username, req.password)
    if not success:
        raise HTTPException(status_code=401, detail=data)
    return {"success": True, "account": data}

@app.get("/account/{username}", tags=["Account"])
async def get_account(username: str):
    account = account_manager.get_account(username)
    if not account:
        raise HTTPException(status_code=404, detail="Account not found.")
    return {"success": True, "account": account}

@app.post("/account/progress", tags=["Account"])
async def update_progress(req: AccountUpdateProgressRequest):
    success, msg = account_manager.update_progress(req.username, req.letter, req.level)
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return {"success": True, "message": msg}
