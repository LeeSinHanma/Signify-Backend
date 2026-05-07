import csv
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Constants
WINDOW_SIZE = 30
FEATURE_COUNT = 63
STATIC_CSV = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"
CALIB_CSV = Path(__file__).resolve().parent / "data" / "raw" / "calibration_landmarks.csv"
MOTION_CSV = Path(__file__).resolve().parent / "data" / "raw" / "motion_landmarks.csv"
MODEL_OUT = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"

def load_raw_frames():
    """Loads individual frames for the static (per-frame) model."""
    features = []
    labels = []
    for csv_path in [STATIC_CSV, CALIB_CSV]:
        if not csv_path.exists(): continue
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    labels.append(row["label"].strip().upper())
                    features.append([float(row[f"f{i}"]) for i in range(FEATURE_COUNT)])
                except: continue
    return features, labels

def load_static_as_sequences(csv_path: Path):
    if not csv_path.exists(): return [], []
    features, labels = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                label = row["label"].strip().upper()
                frame = [float(row[f"f{i}"]) for i in range(FEATURE_COUNT)]
                features.append(frame * WINDOW_SIZE)
                labels.append(label)
            except: continue
    return features, labels

def load_motion_sequences(csv_path: Path):
    if not csv_path.exists(): return [], []
    features, labels = [], []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                labels.append(row["label"].strip().upper())
                features.append([float(row[f"f{i}"]) for i in range(WINDOW_SIZE * FEATURE_COUNT)])
            except: continue
    return features, labels

def main():
    print("--- 1. Training Per-Frame Model (Static) ---")
    f_feat, f_lab = load_raw_frames()
    X_f = np.array(f_feat, dtype=np.float32)
    y_f = np.array(f_lab)
    clf_frame = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    clf_frame.fit(X_f, y_f)
    print(f"Frame model trained on {len(X_f)} samples.")

    print("\n--- 2. Training Windowed Model (Motion) ---")
    s_feat, s_lab = load_static_as_sequences(STATIC_CSV)
    c_feat, c_lab = load_static_as_sequences(CALIB_CSV)
    m_feat, m_lab = load_motion_sequences(MOTION_CSV)
    
    X_w = np.array(s_feat + c_feat + m_feat, dtype=np.float32)
    y_w = np.array(s_lab + c_lab + m_lab)
    
    clf_window = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
    clf_window.fit(X_w, y_w)
    print(f"Window model trained on {len(X_w)} samples.")

    # Save both models
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "frame_model": clf_frame,
        "window_model": clf_window,
        "labels": sorted(set(y_w))
    }, MODEL_OUT)
    print(f"\n✅ Dual-Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    main()
