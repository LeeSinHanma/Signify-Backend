import csv
import argparse
from collections import Counter
from pathlib import Path
import re

import numpy as np

RAW_DIR = Path(__file__).resolve().parent / "data" / "raw"


def get_feature_columns(headers):
    pairs = []
    for h in headers:
        m = re.fullmatch(r"f(\d+)", h)
        if m:
            pairs.append((int(m.group(1)), h))

    pairs.sort(key=lambda x: x[0])
    indices = [i for i, _ in pairs]
    if not indices:
        raise ValueError("CSV has no feature columns (expected f0, f1, ...).")

    expected = list(range(indices[-1] + 1))
    if indices != expected:
        raise ValueError("Feature columns are not contiguous from f0 to fN.")

    return [name for _, name in pairs]


def load_rows(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if "label" not in headers:
            raise ValueError("CSV is missing 'label' column.")
        feature_columns = get_feature_columns(headers)

        rows = []
        for row in reader:
            rows.append(row)
    return rows, feature_columns


def row_to_features(row, feature_columns):
    return np.array([float(row[c]) for c in feature_columns], dtype=np.float32)


def validate_file(csv_path: Path) -> None:
    rows, feature_columns = load_rows(csv_path)
    feature_count = len(feature_columns)
    if not rows:
        raise ValueError("Dataset is empty.")

    class_counts = Counter()
    invalid_labels = []
    parse_errors = 0
    zero_vectors = 0

    feature_rows = []

    for idx, row in enumerate(rows, start=2):
        label = (row.get("label") or "").strip().upper()
        if not label:
            invalid_labels.append((idx, label))
            continue

        try:
            features = row_to_features(row, feature_columns)
        except Exception:
            parse_errors += 1
            continue

        class_counts[label] += 1
        if np.allclose(features, 0.0):
            zero_vectors += 1

        feature_rows.append(features)

    print("=== Dataset Validation Report ===")
    print(f"File: {csv_path}")
    print(f"Feature columns: {feature_count}")
    print(f"Total CSV rows: {len(rows)}")
    print(f"Parsed valid rows: {len(feature_rows)}")
    print()

    print("Class counts:")
    for label in sorted(class_counts.keys()):
        print(f"  {label}: {class_counts.get(label, 0)}")

    print()
    min_count = min(class_counts.values()) if class_counts else 0
    max_count = max(class_counts.values()) if class_counts else 0
    if min_count == 0:
        print("[WARN] At least one class has zero samples.")
    elif max_count > 0 and (max_count / min_count) > 1.5:
        print("[WARN] Class imbalance is high (>1.5x). Try balancing samples.")
    else:
        print("[OK] Class balance looks reasonable.")

    if invalid_labels:
        print(f"[WARN] Invalid labels: {len(invalid_labels)} rows")
        preview = invalid_labels[:10]
        for line_no, label in preview:
            print(f"  line {line_no}: label='{label}'")
        if len(invalid_labels) > len(preview):
            print(f"  ... and {len(invalid_labels) - len(preview)} more")
    else:
        print("[OK] No invalid labels detected.")

    if parse_errors:
        print(f"[WARN] Rows with parse errors: {parse_errors}")
    else:
        print("[OK] No parse errors in feature columns.")

    if zero_vectors:
        print(f"[WARN] Zero-feature rows: {zero_vectors}")
    else:
        print("[OK] No zero-feature rows detected.")

    if feature_rows:
        X = np.stack(feature_rows)

        rounded = np.round(X, 6)
        _, counts = np.unique(rounded, axis=0, return_counts=True)
        duplicate_rows = int(np.sum(counts[counts > 1] - 1))
        duplicate_ratio = duplicate_rows / len(feature_rows)

        print()
        print(f"Duplicate rows (approx): {duplicate_rows}")
        print(f"Duplicate ratio: {duplicate_ratio:.2%}")

        if duplicate_ratio > 0.20:
            print("[WARN] High duplicate ratio (>20%). Add more pose variation.")
        else:
            print("[OK] Duplicate ratio is acceptable.")

        std_mean = float(np.mean(np.std(X, axis=0)))
        print(f"Mean feature std: {std_mean:.6f}")
        if std_mean < 0.01:
            print("[WARN] Very low variation detected. Collect more varied samples.")
        else:
            print("[OK] Feature variation looks healthy.")

    print()
    print("Validation complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate landmark CSV dataset quality.")
    parser.add_argument("--csv", type=Path, help="Path to a single landmark CSV dataset")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Directory of CSV datasets to validate")
    args = parser.parse_args()

    if args.csv is not None:
        validate_file(args.csv)
        return

    csv_files = sorted(args.raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {args.raw_dir}")

    print(f"Validating {len(csv_files)} CSV file(s) in {args.raw_dir}\n")
    for i, csv_path in enumerate(csv_files, start=1):
        print(f"[{i}/{len(csv_files)}] {csv_path.name}")
        validate_file(csv_path)
        print("-" * 60)


if __name__ == "__main__":
    main()
