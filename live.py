import time
import argparse
from collections import deque, Counter
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import joblib
import mediapipe as mp
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"
WINDOW_SIZE = 30
FEATURE_COUNT = 63
MOTION_LETTERS = ['J', 'Z']

def ensure_model_file(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return
    print(f"Downloading model to: {model_path}")
    urlretrieve(MODEL_URL, model_path)

def normalize_landmarks(hand_landmarks):
    wrist = hand_landmarks[0]
    centered = []
    for lm in hand_landmarks:
        centered.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    max_abs = max(abs(v) for v in centered)
    if max_abs == 0: return centered
    return [v / max_abs for v in centered]

def main() -> None:
    parser = argparse.ArgumentParser(description="Live sign prediction with voting consensus.")
    parser.add_argument("--model", type=Path, default=CLASSIFIER_PATH, help="Path to trained model file")
    parser.add_argument("--threshold", type=float, default=0.70, help="Minimum confidence to accept prediction")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"\n[ERROR] Trained model NOT found at: {args.model}")
        print("Please run 'python train_motion.py' first.\n")
        return

    ensure_model_file(LANDMARKER_PATH)
    payload = joblib.load(args.model)
    clf_frame = payload["frame_model"]
    clf_window = payload["window_model"]

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_timestamp_ms = -1
    landmark_history = deque(maxlen=WINDOW_SIZE)

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(time.perf_counter() * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            prediction_text = "No hand"
            color = (0, 0, 255)
            consensus_info = ""

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)
                landmark_history.append(features)

                if len(landmark_history) == WINDOW_SIZE:
                    # 1. Motion Prediction (Window-based)
                    window_flat = []
                    for f in landmark_history: window_flat.extend(f)
                    
                    w_probs = clf_window.predict_proba(np.array([window_flat], dtype=np.float32))[0]
                    w_idx = int(np.argmax(w_probs))
                    w_label = clf_window.classes_[w_idx]
                    w_conf = float(w_probs[w_idx])

                    # 2. Voting Prediction (Frame-based)
                    f_batch = np.array(list(landmark_history), dtype=np.float32)
                    f_preds = clf_frame.predict(f_batch)
                    vote_counts = Counter(f_preds)
                    v_label, v_count = vote_counts.most_common(1)[0]
                    v_conf = v_count / WINDOW_SIZE

                    # 3. Decision Logic
                    if w_label in MOTION_LETTERS and w_conf > 0.85:
                        # Prioritize Motion Model for J and Z
                        prediction_text = f"{w_label} (Motion:{w_conf:.2f})"
                        color = (255, 100, 0)
                    elif v_conf >= args.threshold:
                        # Prioritize Majority Vote for static letters
                        prediction_text = f"{v_label} (Vote:{v_conf:.2f})"
                        color = (0, 220, 0)
                    else:
                        prediction_text = "Analyzing..."
                        color = (0, 165, 255)
                    
                    consensus_info = f"Top Vote: {v_label} ({v_count}/{WINDOW_SIZE})"
                else:
                    prediction_text = f"Buffering... ({len(landmark_history)}/{WINDOW_SIZE})"
                    color = (0, 255, 255)

                # Draw landmarks
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(frame, f"AI: {prediction_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            if consensus_info:
                cv2.putText(frame, consensus_info, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, "Press Q to quit", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            cv2.imshow("Unified Sign Recognition", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
