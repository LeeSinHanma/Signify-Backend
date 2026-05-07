import csv
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"

def add_jitter(X, noise_level=0.005):
    """
    Adds small random noise to landmarks to simulate camera jitter and 
    variations in hand position, making the model more robust.
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    labels = []
    features = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().upper()
            try:
                vector = [float(row[f"f{i}"]) for i in range(63)]
                labels.append(label)
                features.append(vector)
            except (ValueError, KeyError):
                continue

    if not labels:
        raise ValueError("No valid rows found in dataset.")

    return np.array(features, dtype=np.float32), np.array(labels)

def main() -> None:
    parser = argparse.ArgumentParser(description="Optimized training for sign language classifier.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to landmark CSV dataset")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="Path to save trained model")
    parser.add_argument("--augment", action="store_true", default=True, help="Apply data augmentation (jitter)")
    args = parser.parse_args()

    print(f"Loading dataset from {args.csv}...")
    X, y = load_dataset(args.csv)

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
