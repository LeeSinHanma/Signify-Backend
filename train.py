import csv
import argparse
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

CSV_PATH = Path(__file__).resolve().parent / "data" / "raw" / "vowels_from_images_landmarks.csv"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "vowel_random_forest.joblib"


def load_dataset(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    labels = []
    features = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().upper()
            vector = [float(row[f"f{i}"]) for i in range(63)]
            labels.append(label)
            features.append(vector)

    if not labels:
        raise ValueError("No rows found in dataset.")

    return np.array(features, dtype=np.float32), np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train vowel classifier from landmark CSV.")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to landmark CSV dataset")
    parser.add_argument("--model-out", type=Path, default=MODEL_PATH, help="Path to save trained model")
    args = parser.parse_args()

    X, y = load_dataset(args.csv)

    counts = Counter(y.tolist())
    print("Class counts:", dict(counts))

    if len(counts) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    min_class_count = min(counts.values())
    if min_class_count < 5:
        raise ValueError("Each class needs at least 5 samples.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=4))

    unique_labels = sorted(set(y.tolist()))
    cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
    print("Confusion matrix labels:", unique_labels)
    print(cm)

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "labels": unique_labels}, args.model_out)
    print(f"\nSaved model to: {args.model_out}")


if __name__ == "__main__":
    main()
