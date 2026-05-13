import csv
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"
CALIBRATION_CSV_PATH = DATA_DIR / "calibration_landmarks.csv"
MOTION_CSV_PATH = DATA_DIR / "motion_landmarks.csv"
FEATURE_COUNT = 63
WINDOW_SIZE = 30  # Number of frames in a motion sequence

def add_calibration_sample(features: list[float], label: str) -> None:
    """Appends a new calibrated sample to the calibration CSV."""
    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Expected {FEATURE_COUNT} features, got {len(features)}")
    
    label = label.strip().upper()
    if not label:
        raise ValueError("Label cannot be empty")

    CALIBRATION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    file_exists = CALIBRATION_CSV_PATH.exists()
    
    with CALIBRATION_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            # Write header
            header = ["label"] + [f"f{i}" for i in range(FEATURE_COUNT)]
            writer.writerow(header)
        
        row = [label] + features
        writer.writerow(row)

def add_motion_sample(sequence_features: list[float], label: str) -> None:
    """
    Appends a full motion sequence (WINDOW_SIZE * FEATURE_COUNT) as one row.
    sequence_features should be a flat list of all frames.
    """
    expected_size = WINDOW_SIZE * FEATURE_COUNT
    if len(sequence_features) != expected_size:
        raise ValueError(f"Expected {expected_size} features, got {len(sequence_features)}")
    
    label = label.strip().upper()
    MOTION_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = MOTION_CSV_PATH.exists()
    
    with MOTION_CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ["label"] + [f"f{i}" for i in range(expected_size)]
            writer.writerow(header)
        writer.writerow([label] + sequence_features)

def load_csv_data(csv_path: Path):
    """Loads feature and label data from a given CSV."""
    if not csv_path.exists():
        return [], []
        
    features = []
    labels = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lbl = row.get("label", "").strip().upper()
            if not lbl:
                continue
            try:
                vec = [float(row[f"f{i}"]) for i in range(FEATURE_COUNT)]
                features.append(vec)
                labels.append(lbl)
            except Exception:
                continue
    return features, labels

def retrain_model(base_csv_path: Path, model_out_path: Path):
    """
    Retrains the model using base dataset, calibration dataset, and motion sequences.
    Converts static samples into windowed sequences for a unified model.
    """
    import train_motion
    
    print("Loading datasets...")
    
    # 1. Train Per-Frame Model (Static)
    f_feat, f_lab = train_motion.load_raw_frames()
    X_f = np.array(f_feat, dtype=np.float32)
    y_f = np.array(f_lab)
    clf_frame = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)
    if len(X_f) > 0:
        clf_frame.fit(X_f, y_f)
    
    # 2. Train Windowed Model (Motion)
    s_feat, s_lab = train_motion.load_static_as_sequences(base_csv_path)
    c_feat, c_lab = train_motion.load_static_as_sequences(CALIBRATION_CSV_PATH)
    m_feat, m_lab = train_motion.load_motion_sequences(MOTION_CSV_PATH)
    
    all_features = s_feat + c_feat + m_feat
    all_labels = s_lab + c_lab + m_lab
    
    if not all_features:
        raise ValueError("No data available for training.")
        
    X_w = np.array(all_features, dtype=np.float32)
    y_w = np.array(all_labels)
    
    clf_window = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    
    clf_window.fit(X_w, y_w)
    
    # Calculate training accuracy
    y_pred = clf_window.predict(X_w)
    accuracy = accuracy_score(y_w, y_pred)
    
    unique_labels = sorted(set(y_w.tolist()))
    
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "frame_model": clf_frame,
        "window_model": clf_window,
        "labels": unique_labels
    }, model_out_path)
    
    return clf_window, unique_labels, float(accuracy), len(c_lab) + len(m_lab)
