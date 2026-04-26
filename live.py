import time
import argparse
from collections import deque
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
    if max_abs == 0:
        return centered

    return [v / max_abs for v in centered]


def main() -> None:
    parser = argparse.ArgumentParser(description="Live sign prediction with confidence smoothing.")
    parser.add_argument("--model", type=Path, default=CLASSIFIER_PATH, help="Path to trained model file")
    parser.add_argument("--threshold", type=float, default=0.65, help="Minimum confidence to accept prediction")
    parser.add_argument("--smooth-window", type=int, default=6, help="Number of recent frames to smooth")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Trained classifier not found: {args.model}")

    ensure_model_file(LANDMARKER_PATH)
    payload = joblib.load(args.model)
    clf = payload["model"]

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
    prob_history = deque(maxlen=max(1, args.smooth_window))

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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

            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)
                probs = clf.predict_proba(np.array([features], dtype=np.float32))[0]
                prob_history.append(probs)
                avg_probs = np.mean(np.stack(prob_history, axis=0), axis=0)

                pred_idx = int(np.argmax(avg_probs))
                pred_label = clf.classes_[pred_idx]
                pred_conf = float(avg_probs[pred_idx])

                if pred_conf >= args.threshold:
                    prediction_text = f"{pred_label} ({pred_conf:.2f})"
                    color = (0, 220, 0)
                else:
                    prediction_text = f"Uncertain ({pred_conf:.2f})"
                    color = (0, 165, 255)

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(frame, f"Prediction: {prediction_text}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Thr:{args.threshold:.2f} Win:{max(1, args.smooth_window)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, "Press Q to quit", (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.imshow("Live Vowel Prediction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
