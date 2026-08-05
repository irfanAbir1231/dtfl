#!/usr/bin/env python3
"""Download and preprocess PAD-UFES-20 for DTFL.

The generated split is patient-safe: a patient is assigned to exactly one of
train/test, and every training patient is assigned to exactly one federated
client. Runtime image resizing and augmentation remain in the PyTorch loader,
just as they do for HAM10000.
"""

import argparse
import csv
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


KAGGLE_HANDLE = "mahdavi1202/skin-cancer"
EXPECTED_IMAGE_COUNT = 2298
EXPECTED_DIAGNOSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def download_dataset(dataset_root, force=False):
    """Download and extract the complete Kaggle dataset into ``raw``."""
    raw_dir = dataset_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # KaggleHub <1.0 does not support output_dir. Keeping its cache under raw
    # still places the extracted dataset inside data/PAD and avoids a duplicate
    # copy in the user's home directory.
    os.environ["KAGGLEHUB_CACHE"] = str(raw_dir / ".kagglehub")
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required for --download. Install it with "
            "'python3 -m pip install kagglehub'."
        ) from exc

    print(f"Downloading {KAGGLE_HANDLE} into {raw_dir} ...")
    try:
        downloaded_path = kagglehub.dataset_download(
            KAGGLE_HANDLE,
            output_dir=str(raw_dir),
            force_download=force,
        )
    except TypeError:
        # Compatibility with kagglehub 0.3.13 (the newest release supporting
        # Python 3.9). KAGGLEHUB_CACHE above controls its destination.
        downloaded_path = kagglehub.dataset_download(
            KAGGLE_HANDLE,
            force_download=force,
        )
    print(f"KaggleHub dataset path: {downloaded_path}")
    return Path(downloaded_path)


def discover_metadata(dataset_root):
    candidates = [
        path
        for path in dataset_root.rglob("metadata.csv")
        if "processed" not in path.parts
    ]
    if not candidates:
        raise FileNotFoundError(
            f"metadata.csv was not found under '{dataset_root}'. Run this "
            "script with --download or place the extracted Kaggle files in "
            f"'{dataset_root / 'raw'}'."
        )
    if len(candidates) > 1:
        locations = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"Multiple metadata.csv files found: {locations}")
    return candidates[0]


def discover_images(dataset_root):
    images_by_name = {}
    for image_path in dataset_root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if "processed" in image_path.parts:
            continue
        previous = images_by_name.get(image_path.name)
        if previous is not None and previous != image_path:
            raise RuntimeError(
                f"Duplicate PAD image filename '{image_path.name}' found at "
                f"'{previous}' and '{image_path}'."
            )
        images_by_name[image_path.name] = image_path
    return images_by_name


def read_metadata(metadata_path):
    with metadata_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata CSV has no header: '{metadata_path}'.")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    required = {"patient_id", "lesion_id", "diagnostic", "img_id"}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"Metadata CSV '{metadata_path}' is missing columns: {sorted(missing)}"
        )
    return fieldnames, rows


def preprocess_metadata(dataset_root, strict=True):
    metadata_path = discover_metadata(dataset_root)
    fieldnames, rows = read_metadata(metadata_path)
    images_by_name = discover_images(dataset_root)

    if strict and len(rows) != EXPECTED_IMAGE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_IMAGE_COUNT} PAD metadata rows, found {len(rows)}."
        )

    diagnoses = sorted({row["diagnostic"].strip().upper() for row in rows})
    if strict and tuple(diagnoses) != EXPECTED_DIAGNOSES:
        raise ValueError(
            f"Expected PAD diagnoses {list(EXPECTED_DIAGNOSES)}, found {diagnoses}."
        )
    label_mapping = {diagnosis: index for index, diagnosis in enumerate(diagnoses)}

    numeric_ages = []
    for row in rows:
        try:
            numeric_ages.append(float(row.get("age", "")))
        except (TypeError, ValueError):
            pass
    median_age = statistics.median(numeric_ages) if numeric_ages else None

    processed_rows = []
    missing_images = []
    seen_image_ids = set()
    for original in rows:
        row = dict(original)
        image_id = row["img_id"].strip()
        if image_id in seen_image_ids:
            raise ValueError(f"Duplicate img_id in PAD metadata: '{image_id}'.")
        seen_image_ids.add(image_id)

        diagnosis = row["diagnostic"].strip().upper()
        image_path = images_by_name.get(image_id)
        if image_path is None:
            missing_images.append(image_id)
            continue

        if median_age is not None and not row.get("age", "").strip():
            row["age"] = f"{median_age:g}"
        row["diagnostic"] = diagnosis
        row["label"] = str(label_mapping[diagnosis])
        row["path"] = image_path.relative_to(dataset_root).as_posix()
        processed_rows.append(row)

    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise FileNotFoundError(
            f"Could not find {len(missing_images)} PAD images referenced by metadata. "
            f"First missing image IDs: {preview}"
        )
    if len(images_by_name) != len(processed_rows):
        unused = sorted(set(images_by_name).difference(seen_image_ids))
        raise ValueError(
            f"Found {len(images_by_name)} PAD images but {len(processed_rows)} metadata "
            f"rows. Unreferenced image examples: {unused[:10]}"
        )

    output_fields = list(fieldnames)
    for field in ("label", "path"):
        if field not in output_fields:
            output_fields.append(field)
    return output_fields, processed_rows, label_mapping, median_age


def select_patient_split(rows, test_fraction=0.2, seed=42, attempts=3000):
    """Select a deterministic patient-disjoint split with low class drift."""
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    patient_ids = sorted(patients)
    test_patient_count = max(1, round(len(patient_ids) * test_fraction))
    total_by_class = Counter(row["label"] for row in rows)
    target_test_size = len(rows) * test_fraction
    target_by_class = {
        label: count * test_fraction for label, count in total_by_class.items()
    }

    best = None
    for attempt in range(attempts):
        shuffled = list(patient_ids)
        random.Random(seed + attempt).shuffle(shuffled)
        test_patients = set(shuffled[:test_patient_count])
        test_rows = [
            row for patient_id in test_patients for row in patients[patient_id]
        ]
        test_by_class = Counter(row["label"] for row in test_rows)
        train_by_class = total_by_class - test_by_class

        missing_penalty = sum(
            test_by_class[label] == 0 or train_by_class[label] == 0
            for label in total_by_class
        )
        size_error = abs(len(test_rows) - target_test_size) / max(len(rows), 1)
        class_error = sum(
            abs(test_by_class[label] - target_by_class[label])
            / max(target_by_class[label], 1.0)
            for label in total_by_class
        )
        score = missing_penalty * 1000.0 + size_error * 5.0 + class_error
        candidate = (score, sorted(test_patients))
        if best is None or candidate < best:
            best = candidate

    test_patients = set(best[1])
    train_rows = [row for row in rows if row["patient_id"] not in test_patients]
    test_rows = [row for row in rows if row["patient_id"] in test_patients]
    if {row["patient_id"] for row in train_rows}.intersection(
        row["patient_id"] for row in test_rows
    ):
        raise RuntimeError("Patient leakage detected between PAD train and test splits.")
    return train_rows, test_rows


def _dirichlet(size, alpha, rng):
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(draws)
    return [draw / total for draw in draws]


def _client_partition_penalty(client_rows, labels):
    sizes = [len(rows) for rows in client_rows]
    average_size = sum(sizes) / len(sizes)
    minimum_size = min(sizes)
    size_ratio = max(sizes) / max(minimum_size, 1)
    coverage = {
        label: sum(any(row["label"] == label for row in rows) for rows in client_rows)
        for label in labels
    }
    classes_per_client = [len({row["label"] for row in rows}) for rows in client_rows]

    penalty = 0.0
    penalty += max(0.0, average_size * 0.35 - minimum_size) * 20.0
    penalty += max(0.0, size_ratio - 4.0) * 100.0
    penalty += sum(max(0, 3 - count) * 100.0 for count in classes_per_client)
    penalty += sum(max(0, 3 - count) * 100.0 for count in coverage.values())
    return penalty, sizes, coverage, classes_per_client


def partition_training_patients(rows, client_count=8, alpha=0.5, seed=42, attempts=500):
    """Build deterministic, patient-disjoint, label-skewed client shards."""
    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    if client_count < 1 or client_count > len(patients):
        raise ValueError(
            f"client_count must be between 1 and {len(patients)}, got {client_count}."
        )

    labels = sorted({row["label"] for row in rows}, key=int)
    total_by_class = Counter(row["label"] for row in rows)
    patient_items = list(patients.items())
    best = None

    for attempt in range(attempts):
        rng = random.Random(seed + 10_000 + attempt)
        desired_by_class = {
            label: [
                share * total_by_class[label]
                for share in _dirichlet(client_count, alpha, rng)
            ]
            for label in labels
        }
        capacity_weights = [rng.uniform(0.8, 1.2) for _ in range(client_count)]
        capacity_total = sum(capacity_weights)
        desired_sizes = [
            len(rows) * weight / capacity_total for weight in capacity_weights
        ]

        rng.shuffle(patient_items)
        ordered_patients = sorted(patient_items, key=lambda item: len(item[1]), reverse=True)
        client_rows = [[] for _ in range(client_count)]
        client_class_counts = [Counter() for _ in range(client_count)]

        for _, patient_rows in ordered_patients:
            patient_counts = Counter(row["label"] for row in patient_rows)
            scores = []
            for client_id in range(client_count):
                class_score = sum(
                    patient_counts[label]
                    * (
                        desired_by_class[label][client_id]
                        - client_class_counts[client_id][label]
                    )
                    / max(desired_by_class[label][client_id], 1.0)
                    for label in patient_counts
                ) / max(len(patient_rows), 1)
                size_score = (
                    desired_sizes[client_id] - len(client_rows[client_id])
                ) / max(desired_sizes[client_id], 1.0)
                scores.append((2.0 * class_score + size_score, rng.random(), client_id))
            selected = max(scores)[2]
            client_rows[selected].extend(patient_rows)
            client_class_counts[selected].update(patient_counts)

        penalty, sizes, coverage, classes_per_client = _client_partition_penalty(
            client_rows, labels
        )
        class_skew = sum(
            max(counts[label] for counts in client_class_counts)
            / max(total_by_class[label], 1)
            for label in labels
        )
        candidate_score = penalty - class_skew
        candidate = (
            candidate_score,
            [sorted(client, key=lambda row: row["img_id"]) for client in client_rows],
            sizes,
            coverage,
            classes_per_client,
        )
        if best is None or candidate_score < best[0]:
            best = candidate
        if penalty == 0:
            break

    client_rows = best[1]
    assigned_images = [row["img_id"] for shard in client_rows for row in shard]
    if len(assigned_images) != len(set(assigned_images)):
        raise RuntimeError("A PAD image was assigned to more than one client.")
    if set(assigned_images) != {row["img_id"] for row in rows}:
        raise RuntimeError("PAD client shards do not cover the complete training split.")

    patient_owners = {}
    for client_id, shard in enumerate(client_rows):
        for patient_id in {row["patient_id"] for row in shard}:
            if patient_id in patient_owners:
                raise RuntimeError(
                    f"Patient '{patient_id}' was assigned to multiple PAD clients."
                )
            patient_owners[patient_id] = client_id
    return client_rows


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def prepare_dataset(
    dataset_root,
    client_count=8,
    test_fraction=0.2,
    alpha=0.5,
    seed=42,
    strict=True,
):
    fieldnames, rows, label_mapping, median_age = preprocess_metadata(
        dataset_root, strict=strict
    )
    train_rows, test_rows = select_patient_split(
        rows, test_fraction=test_fraction, seed=seed
    )
    client_rows = partition_training_patients(
        train_rows, client_count=client_count, alpha=alpha, seed=seed
    )

    processed_dir = dataset_root / "processed"
    write_csv(processed_dir / "PAD_metadata_preprocessed.csv", fieldnames, rows)
    write_csv(processed_dir / "PAD_metadata_train.csv", fieldnames, train_rows)
    write_csv(processed_dir / "PAD_metadata_test.csv", fieldnames, test_rows)
    for client_id, shard in enumerate(client_rows, start=1):
        write_csv(processed_dir / f"client_{client_id}_train.csv", fieldnames, shard)

    summary = {
        "dataset": "PAD-UFES-20",
        "source": KAGGLE_HANDLE,
        "seed": seed,
        "test_fraction": test_fraction,
        "partition_alpha": alpha,
        "label_mapping": label_mapping,
        "median_age": median_age,
        "total_images": len(rows),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "total_patients": len({row["patient_id"] for row in rows}),
        "train_patients": len({row["patient_id"] for row in train_rows}),
        "test_patients": len({row["patient_id"] for row in test_rows}),
        "train_class_counts": dict(Counter(row["diagnostic"] for row in train_rows)),
        "test_class_counts": dict(Counter(row["diagnostic"] for row in test_rows)),
        "client_images": [len(shard) for shard in client_rows],
        "client_patients": [
            len({row["patient_id"] for row in shard}) for shard in client_rows
        ],
        "client_class_counts": [
            dict(Counter(row["diagnostic"] for row in shard)) for shard in client_rows
        ],
    }
    summary_path = processed_dir / "PAD_preprocessing_summary.json"
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_summary.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary_summary, summary_path)

    print("PAD-UFES-20 preprocessing complete")
    print(f"  Images: {len(rows)} total, {len(train_rows)} train, {len(test_rows)} test")
    print(
        "  Patients: "
        f"{summary['total_patients']} total, {summary['train_patients']} train, "
        f"{summary['test_patients']} test"
    )
    print(f"  Label mapping: {label_mapping}")
    for client_id, shard in enumerate(client_rows, start=1):
        counts = dict(sorted(Counter(row["diagnostic"] for row in shard).items()))
        print(f"  Client {client_id}: {len(shard)} images | {counts}")
    print(f"  Output: {processed_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and preprocess PAD-UFES-20 for DTFL."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="PAD dataset directory (default: directory containing this script).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the complete Kaggle dataset before preprocessing.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload and overwrite the Kaggle files.",
    )
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--partition-alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a nonstandard image count (intended only for development tests).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    if args.download or args.force_download:
        download_dataset(dataset_root, force=args.force_download)
    prepare_dataset(
        dataset_root,
        client_count=args.clients,
        test_fraction=args.test_fraction,
        alpha=args.partition_alpha,
        seed=args.seed,
        strict=not args.allow_partial,
    )


if __name__ == "__main__":
    main()
