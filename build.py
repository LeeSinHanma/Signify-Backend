import argparse
import csv
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "hand_landmarker.task"
DEFAULT_IMAGES_DIR = Path(__file__).resolve().parent / "data" / "images"
DEFAULT_OUTPUT_CSV = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def collect_image_files(label_dir: Path):
    files = []
    for path in sorted(label_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            files.append(path)
    return files


def write_header_if_needed(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return

    headers = ["label", "source_path"] + [f"f{i}" for i in range(63)]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)


def load_processed_paths(csv_path: Path) -> set:
    """Load all previously processed image paths from CSV."""
    processed = set()
    if not csv_path.exists():
        return processed
    
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source_path = row.get("source_path", "").strip()
                if source_path:
                    processed.add(source_path)
    except Exception:
        pass
    
    return processed



def get_available_labels(images_dir: Path) -> list:
    """Discover available labels by scanning subdirectories."""
    labels = []
    if not images_dir.exists():
        return labels
    
    for item in sorted(images_dir.iterdir()):
        if item.is_dir():
            labels.append(item.name.upper())
    
    return labels

def main() -> None:
    parser = argparse.ArgumentParser(description="Build landmark CSV dataset from labeled vowel image folders (incremental by default).")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Root image folder with A/E/I/O/U subfolders")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Output CSV path")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="MediaPipe hand landmarker .task path")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild from scratch (delete and recreate CSV)")
    arDiscover available labels dynamically
    available_labels = get_available_labels(args.images_dir)
    if not available_labels:
        print(f"No label folders found in {args.images_dir}")
        print("Create folders like: data/images/A, data/images/B, etc.")
        return
    
    print(f"Found labels: {', '.join(available_labels)}")
    
    # Load already-processed image paths
    processed_paths = load_processed_paths(args.output_csv)
    print(f"Previously processed: {len(processed_paths)} images")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(args.model)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    total_new = 0
    total_skipped_existing = 0
    total_failed = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        with args.output_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for label in available_labels
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    total_new = 0
    total_skipped_existing = 0
    total_failed = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        with args.output_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            for label in VALID_LABELS:
                label_dir = args.images_dir / label
                image_files = collect_image_files(label_dir) if label_dir.exists() else []
                print(f"\nLabel {label}: found {len(image_files)} image(s)")

                label_new = 0
                label_existing = 0
                label_failed = 0

                for image_path in image_files:
                    image_path_str = str(image_path)
                    
                    # Skip if already processed
                    if image_path_str in processed_paths:
                        label_existing += 1
                        continue

                    bgr = cv2.imread(image_path_str)
                    if bgr is None:
                        label_failed += 1
                        continue

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    result = landmarker.detect(mp_image)
                    if not result.hand_landmarks:
                        label_failed += 1
                        continue

                    features = normalize_landmarks(result.hand_landmarks[0])
                    row = [label, image_path_str] + [f"{v:.8f}" for v in features]
                    writer.writerow(row)
                    label_new += 1

                total_new += label_new
                total_skipped_existing += label_existing
                total_failed += label_failed
                print(f"  new={label_new}, already_processed={label_existing}, failed={label_failed}")

    print("\n" + "="*60)
    print("Build complete!")
    print(f"Output CSV: {args.output_csv}")
    print(f"New rows added: {total_new}")
    print(f"Already processed (skipped): {total_skipped_existing}")
    print(f"Failed (no hand/read error): {total_failed}")
    print(f"Total in CSV now: {len(processed_paths) + total_new}")
    print("="*60)
    print("\nUsage tips:")
    print("  Normal incremental run: python build.py")
    print("  Rebuild from scratch:   python build.py --rebuild")


if __name__ == "__main__":
    main()
