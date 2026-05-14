import csv
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

CSV_DIR = Path(__file__).resolve().parent / "data" / "raw"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "alphabet_random_forest.joblib"

def add_jitter(X, noise_level=0.005):
    """
    Adds small random noise to landmarks to simulate camera jitter and 
    variations in hand position, making the model more robust.
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def load_dataset(csv_dir: Path):
    """
    Loads and merges all CSV files from the specified directory.
    Combines: alphabet_landmarks.csv, calibration_landmarks.csv, motion_landmarks.csv
    """
    if not csv_dir.exists():
        raise FileNotFoundError(f"Directory not found: {csv_dir}")

    labels = []
    features = []
    
    # Find all CSV files in the directory
    csv_files = sorted(csv_dir.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")
    
    print(f"\nFound {len(csv_files)} CSV file(s) to merge:")
    for csv_file in csv_files:
        print(f"  - {csv_file.name}")
    
    # Load and merge all CSVs
    for csv_file in csv_files:
        print(f"\nLoading {csv_file.name}...")
        file_count = 0
        
        try:
            with csv_file.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    label = row.get("label", "").strip().upper()
                    if not label:
                        continue
                    try:
                        vector = [float(row[f"f{i}"]) for i in range(63)]
                        labels.append(label)
                        features.append(vector)
                        file_count += 1
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f"  Warning: Error reading {csv_file.name}: {e}")
            continue
        
        print(f"  Loaded {file_count} samples from {csv_file.name}")

    if not labels:
        raise ValueError("No valid rows found in any dataset.")

    print(f"\nTotal merged samples: {len(labels)}")
    return np.array(features, dtype=np.float32), np.array(labels)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train sign language classifier by merging all CSVs in data/raw/.")
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR, help="Directory containing CSV files to merge and train on")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply data augmentation (jitter)")
    args = parser.parse_args()

    print(f"Loading and merging datasets from {args.csv_dir}...")
    X, y = load_dataset(args.csv_dir)

    counts = Counter(y.tolist())
    print("Base class counts:", dict(counts))

    # Apply Data Augmentation
    if args.augment:
        print("Applying feature augmentation (jittering)...")
        X_aug = add_jitter(X)
        X = np.vstack((X, X_aug))
        y = np.concatenate((y, y))
        print(f"Augmented dataset size: {len(X)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nSearching for optimal hyperparameters...")
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 'log2']
    }

    base_clf = RandomForestClassifier(random_state=42, class_weight="balanced", n_jobs=-1)
    
    # Quick random search to optimize the forest
    random_search = RandomizedSearchCV(
        base_clf, param_distributions=param_dist, 
        n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1
    )
    random_search.fit(X_train, y_train)
    
    clf = random_search.best_estimator_
    print(f"Best parameters found: {random_search.best_params_}")

    # Cross-Validation Score
    print("\nPerforming 5-fold cross-validation...")
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5)
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Final Training evaluation
    y_pred = clf.predict(X_test)
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    unique_labels = sorted(set(y.tolist()))
    print("\nConfusion Matrix Labels:", unique_labels)
    print(confusion_matrix(y_test, y_pred, labels=unique_labels))

    # Save model
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": unique_labels}, args.model_out)
    print(f"\n[OK] Optimized model saved to: {args.model_out}")

if __name__ == "__main__":
    main()
