import csv
import time
from collections import Counter
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
VOWELS = ["A", "E", "I", "O", "U"]
CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "vowels_landmarks.csv"
MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"


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


def read_counts(csv_path: Path) -> Counter:
    counts = Counter()
    if not csv_path.exists():
        return counts

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "")
            if label in VOWELS:
                counts[label] += 1
    return counts


def ensure_csv(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    headers = ["label", "timestamp_ms"] + [f"f{i}" for i in range(63)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def save_sample(csv_path: Path, label: str, features) -> None:
    timestamp_ms = int(time.time() * 1000)
    row = [label, timestamp_ms] + [f"{v:.8f}" for v in features]
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def main() -> None:
    ensure_model_file(MODEL_PATH)
    ensure_csv(CSV_PATH)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
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
    counts = read_counts(CSV_PATH)

    print("Controls:")
    print("  Press A/E/I/O/U to save a sample for that label")
    print("  Press Q to quit")

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

            features = None
            if result.hand_landmarks:
                hand_landmarks = result.hand_landmarks[0]
                features = normalize_landmarks(hand_landmarks)

                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            cv2.putText(frame, "Collect vowels: A E I O U", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
            cv2.putText(frame, "Press key for label | Q quit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            cv2.putText(frame, f"Counts A:{counts['A']} E:{counts['E']} I:{counts['I']} O:{counts['O']} U:{counts['U']}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            if features is None:
                cv2.putText(frame, "No hand detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
            else:
                cv2.putText(frame, "Hand detected", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)

            cv2.imshow("Collect Vowel Data", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            key_map = {
                ord("a"): "A", ord("A"): "A",
                ord("e"): "E", ord("E"): "E",
                ord("i"): "I", ord("I"): "I",
                ord("o"): "O", ord("O"): "O",
                ord("u"): "U", ord("U"): "U",
            }

            if key in key_map and features is not None:
                label = key_map[key]
                save_sample(CSV_PATH, label, features)
                counts[label] += 1
                print(f"Saved {label}: total {counts[label]}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
