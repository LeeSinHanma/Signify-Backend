import time
import cv2
import mediapipe as mp
from pathlib import Path
import calibrator
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
LANDMARKER_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"
BASE_CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"

# Number of samples to capture in a burst when SPACE is pressed
SAMPLES_PER_BURST = 30
MOTION_LETTERS = ['J', 'Z']

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
    if not LANDMARKER_PATH.exists():
        print(f"Model missing: {LANDMARKER_PATH}")
        # We might need to ensure model file here if build.py hasn't been run
        # but usually it exists in the workspace.
    
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
    
    capturing = False
    samples_collected = 0
    capture_buffer = []  # To store the sequence of frames
    calibrated_any = False
    target_letter = ""

    print("Starting Interactive Calibrator...")
    print("1. Press a letter key (A-Z) to select the sign you want to train.")
    print("2. Press SPACE to start the capture burst.")
    print("3. Press Q to quit and retrain.")

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

                # Draw landmarks
                h, w, _ = frame.shape
                for lm in hand_landmarks:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Handle UI and logic
            if capturing:
                # Capture Stage
                cv2.putText(frame, f"CAPTURING '{target_letter}'...", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                cv2.putText(frame, f"Progress: {samples_collected}/{SAMPLES_PER_BURST}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if features is not None:
                    try:
                        capture_buffer.extend(features)
                        samples_collected += 1
                        calibrated_any = True
                        time.sleep(0.02) 
                    except Exception as e:
                        print(f"Error saving sample: {e}")

                if samples_collected >= SAMPLES_PER_BURST:
                    capturing = False
                    if target_letter in MOTION_LETTERS:
                        calibrator.add_motion_sample(capture_buffer, target_letter)
                        print(f"Saved motion sequence for '{target_letter}'")
                    else:
                        # For static letters, we can save individual frames or the whole sequence
                        # To keep compatibility with existing train.py, we save individual frames
                        for i in range(0, len(capture_buffer), 63):
                            calibrator.add_calibration_sample(capture_buffer[i:i+63], target_letter)
                    
                    samples_collected = 0
                    capture_buffer = []
            
            elif target_letter != "":
                # Confirmation Stage (Waiting for SPACE)
                cv2.putText(frame, f"TARGET: {target_letter}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3)
                cv2.putText(frame, "Press SPACE to start burst", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Or press another letter to change", (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            else:
                # Selection Stage (Waiting for Letter)
                cv2.putText(frame, "SELECT A LETTER (A-Z)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                cv2.putText(frame, "Press key on keyboard...", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Hand detection status
            if not capturing:
                if features is None:
                    cv2.putText(frame, "Hand NOT detected", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Hand detected - Ready", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Live Calibration", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key in (ord("q"), ord("Q")):
                break
            
            # Key Handling
            if not capturing and key != 255:
                char = chr(key).upper()
                if char.isalpha() and len(char) == 1:
                    target_letter = char
                    print(f"Selected target: '{target_letter}'")
                
                if key == ord(' ') and target_letter != "":
                    if features is not None:
                        capturing = True
                        samples_collected = 0
                        capture_buffer = []
                        print(f"Starting burst for '{target_letter}'...")
                    else:
                        print("Cannot start: Hand not detected.")

    cap.release()
    cv2.destroyAllWindows()
    
    print("\n--- Calibration Session Ended ---")
    if calibrated_any:
        print("Retraining model with your new data... Please wait.")
        try:
            clf, unique_labels, acc, used = calibrator.retrain_model(BASE_CSV_PATH, CLASSIFIER_PATH)
            print(f"Success! Model retrained using {used} total personal samples.")
            print(f"Training accuracy: {acc:.2f}")
        except Exception as e:
            print(f"Failed to retrain model: {e}")
    else:
        print("No new data was collected. Model was not retrained.")

if __name__ == "__main__":
    main()
