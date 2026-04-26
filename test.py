import cv2
import mediapipe as mp
import time
from pathlib import Path
from urllib.request import urlretrieve

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def ensure_model_file(model_path: Path) -> None:
    if model_path.exists() and model_path.stat().st_size > 0:
        return

    print(f"Downloading model to: {model_path}")
    urlretrieve(MODEL_URL, model_path)

model_path = Path(__file__).resolve().parent / "hand_landmarker.task"
ensure_model_file(model_path)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(model_path)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions and device availability.")

last_timestamp_ms = -1

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

        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                h, w, _ = frame.shape
                points = []
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))
                    cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                for start_idx, end_idx in HAND_CONNECTIONS:
                    start_pt = points[start_idx]
                    end_pt = points[end_idx]
                    cv2.line(frame, start_pt, end_pt, (255, 180, 0), 2)
            cv2.putText(frame, f"Hands: {len(result.hand_landmarks)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 220, 0), 2)
        else:
            cv2.putText(frame, "Hands: 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()