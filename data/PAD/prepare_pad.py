#!/usr/bin/env python3
"""Download and preprocess PAD-UFES-20 for DTFL.

The generated split is patient-safe: a patient is assigned to exactly one of
train/test, and every training patient is assigned to exactly one federated
client. Runtime image resizing and augmentation remain in the PyTorch loader,
just as they do for HAM10000.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


KAGGLE_HANDLE = "mahdavi1202/skin-cancer"
EXPECTED_IMAGE_COUNT = 2298
EXPECTED_DIAGNOSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
STATISTICS_IMAGE_SIZE = 64


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

    for field in ("patient_id", "lesion_id", "diagnostic", "img_id"):
        blank_rows = [index + 2 for index, row in enumerate(rows) if not row[field].strip()]
        if blank_rows:
            raise ValueError(
                f"PAD metadata has blank '{field}' values at CSV rows {blank_rows[:10]}."
            )

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

    diagnoses_by_lesion = defaultdict(set)
    for row in processed_rows:
        diagnoses_by_lesion[(row["patient_id"], row["lesion_id"])].add(
            row["diagnostic"]
        )
    conflicting_lesions = [
        lesion for lesion, values in diagnoses_by_lesion.items() if len(values) > 1
    ]
    if conflicting_lesions:
        raise ValueError(
            "PAD metadata assigns multiple diagnoses to the same patient/lesion: "
            f"{conflicting_lesions[:10]}"
        )

    output_fields = list(fieldnames)
    for field in ("label", "path"):
        if field not in output_fields:
            output_fields.append(field)
    return output_fields, processed_rows, label_mapping, median_age


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as image_file:
        while True:
            chunk = image_file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def remove_exact_duplicates(dataset_root, rows):
    """Remove exact duplicates, including contradictory duplicate labels.

    PAD-UFES-20 contains byte-identical files under different image IDs. If an
    identical image has conflicting diagnoses, no image-level label can be
    selected safely, so the complete conflict group is excluded. For a
    same-diagnosis group, one deterministic representative is retained.
    """
    rows_by_digest = defaultdict(list)
    for row in rows:
        rows_by_digest[_sha256(dataset_root / row["path"])].append(row)

    retained = []
    duplicate_groups = []
    conflicting_groups = []
    excluded_ids = set()
    for digest, group in sorted(rows_by_digest.items()):
        ordered = sorted(group, key=lambda row: row["img_id"])
        if len(ordered) == 1:
            retained.extend(ordered)
            continue

        diagnoses = sorted({row["diagnostic"] for row in ordered})
        details = {
            "sha256": digest,
            "diagnoses": diagnoses,
            "image_ids": [row["img_id"] for row in ordered],
        }
        if len(diagnoses) > 1:
            conflicting_groups.append(details)
            excluded_ids.update(row["img_id"] for row in ordered)
        else:
            duplicate_groups.append(details)
            retained.append(ordered[0])
            excluded_ids.update(row["img_id"] for row in ordered[1:])

    retained.sort(key=lambda row: row["img_id"])
    report = {
        "exact_duplicate_groups": len(duplicate_groups) + len(conflicting_groups),
        "same_label_duplicate_groups": len(duplicate_groups),
        "conflicting_duplicate_groups": len(conflicting_groups),
        "excluded_images": len(excluded_ids),
        "same_label_duplicates_removed": sum(
            len(group["image_ids"]) - 1 for group in duplicate_groups
        ),
        "conflicting_images_removed": sum(
            len(group["image_ids"]) for group in conflicting_groups
        ),
        "conflicts": conflicting_groups,
    }
    return retained, report


def _image_source(row):
    """Return the archive/acquisition partition used for split balancing."""
    for part in Path(row["path"]).parts:
        if part.startswith("imgs_part_"):
            return part
    return "unknown"


def compute_channel_statistics(dataset_root, rows, image_size=STATISTICS_IMAGE_SIZE):
    """Compute per-channel statistics after deterministic RGB resizing."""
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    pixels = 0
    for row in rows:
        image_path = dataset_root / row["path"]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(
                    (image_size, image_size), Image.Resampling.BILINEAR
                )
                values = np.asarray(image, dtype=np.float64) / 255.0
        except Exception as exc:
            raise ValueError(f"PAD image cannot be decoded: '{image_path}'.") from exc
        channel_sum += values.sum(axis=(0, 1))
        channel_squared_sum += np.square(values).sum(axis=(0, 1))
        pixels += values.shape[0] * values.shape[1]

    mean = channel_sum / pixels
    variance = np.maximum(channel_squared_sum / pixels - np.square(mean), 0.0)
    std = np.sqrt(variance)
    return {
        "image_size": image_size,
        "mean": [float(value) for value in mean],
        "std": [float(value) for value in std],
        "images": len(rows),
    }


def verify_image_files(dataset_root, rows):
    """Fully verify that held-out images can be decoded before writing splits."""
    for row in rows:
        image_path = dataset_root / row["path"]
        try:
            with Image.open(image_path) as image:
                image.verify()
        except Exception as exc:
            raise ValueError(f"PAD image cannot be decoded: '{image_path}'.") from exc


def select_patient_split(rows, test_fraction=0.2, seed=42, attempts=10000):
    """Select a deterministic patient-disjoint, source-aware split.

    Diagnosis alone is insufficient for PAD: the three archive image parts
    have visibly different acquisition/metadata patterns and are strongly
    correlated with diagnosis. The search therefore balances class totals,
    class-by-source totals, and class-bearing patient totals.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be between 0 and 1, got {test_fraction}.")
    if attempts < 1:
        raise ValueError(f"attempts must be positive, got {attempts}.")

    patients = defaultdict(list)
    for row in rows:
        patients[row["patient_id"]].append(row)
    patient_ids = sorted(patients)
    test_patient_count = max(1, round(len(patient_ids) * test_fraction))
    total_by_class = Counter(row["label"] for row in rows)
    total_by_class_source = Counter(
        (row["label"], _image_source(row)) for row in rows
    )
    total_patients_by_class = Counter()
    for patient_rows in patients.values():
        total_patients_by_class.update({row["label"] for row in patient_rows})
    target_test_size = len(rows) * test_fraction
    target_by_class = {
        label: count * test_fraction for label, count in total_by_class.items()
    }
    target_by_class_source = {
        key: count * test_fraction for key, count in total_by_class_source.items()
    }
    target_patients_by_class = {
        label: count * test_fraction for label, count in total_patients_by_class.items()
    }

    best = None
    for attempt in range(attempts):
        test_patients = set(
            random.Random(seed + attempt).sample(patient_ids, test_patient_count)
        )
        test_rows = [
            row for patient_id in test_patients for row in patients[patient_id]
        ]
        test_by_class = Counter(row["label"] for row in test_rows)
        test_by_class_source = Counter(
            (row["label"], _image_source(row)) for row in test_rows
        )
        test_patients_by_class = Counter()
        for patient_id in test_patients:
            test_patients_by_class.update(
                {row["label"] for row in patients[patient_id]}
            )
        train_by_class = total_by_class - test_by_class

        missing_penalty = sum(
            test_by_class[label] == 0 or train_by_class[label] == 0
            for label in total_by_class
        )
        size_error = abs(len(test_rows) - target_test_size) / max(target_test_size, 1)
        class_error = sum(
            abs(test_by_class[label] - target_by_class[label])
            / max(target_by_class[label], 1.0)
            for label in total_by_class
        )
        source_error = sum(
            abs(test_by_class_source[key] - target_by_class_source[key])
            for key in total_by_class_source
        ) / max(target_test_size, 1.0)
        relative_source_error = sum(
            abs(test_by_class_source[key] - target_by_class_source[key])
            / max(target_by_class_source[key], 1.0)
            for key in total_by_class_source
        )
        patient_class_error = sum(
            abs(test_patients_by_class[label] - target_patients_by_class[label])
            / max(target_patients_by_class[label], 1.0)
            for label in total_patients_by_class
        )
        score = (
            missing_penalty * 1000.0
            + size_error * 5.0
            + class_error
            + source_error * 3.0
            + relative_source_error * 0.5
            + patient_class_error * 0.5
        )
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
    total_by_class = Counter(row["label"] for rows in client_rows for row in rows)
    class_counts = [Counter(row["label"] for row in rows) for rows in client_rows]
    coverage = {
        label: sum(counts[label] > 0 for counts in class_counts)
        for label in labels
    }
    classes_per_client = [len({row["label"] for row in rows}) for rows in client_rows]

    penalty = 0.0
    penalty += max(0.0, average_size * 0.65 - minimum_size) * 50.0
    penalty += max(0.0, size_ratio - 1.75) * 500.0
    penalty += sum(max(0, len(labels) - count) * 500.0 for count in classes_per_client)
    penalty += sum(
        max(0, len(client_rows) - count) * 500.0 for count in coverage.values()
    )
    for label in labels:
        # Do not let a rare diagnosis collapse to a single example on a client.
        feasible_minimum = min(3, total_by_class[label] // (2 * len(client_rows)))
        penalty += sum(
            max(0, feasible_minimum - counts[label]) * 100.0
            for counts in class_counts
        )
        # Alpha=0.5 should remain non-IID, but one client should not own most
        # of a diagnosis (especially a rare diagnosis such as melanoma).
        maximum_count = max(3, math.ceil(total_by_class[label] * 0.30))
        penalty += sum(
            max(0, counts[label] - maximum_count) * 100.0
            for counts in class_counts
        )
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
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    if attempts < 1:
        raise ValueError(f"attempts must be positive, got {attempts}.")

    labels = sorted({row["label"] for row in rows}, key=int)
    total_by_class = Counter(row["label"] for row in rows)
    best = None

    for attempt in range(attempts):
        rng = random.Random(seed + 10_000 + attempt)
        # Blend the requested Dirichlet draw with an IID floor. This preserves
        # useful non-IID label skew without producing clients that have only
        # one example of an already rare diagnosis.
        skew_strength = 1.0 / (1.0 + alpha)
        desired_by_class = {
            label: [
                (
                    skew_strength * share
                    + (1.0 - skew_strength) / client_count
                )
                * total_by_class[label]
                for share in _dirichlet(client_count, alpha, rng)
            ]
            for label in labels
        }
        capacity_weights = [rng.uniform(0.95, 1.05) for _ in range(client_count)]
        capacity_total = sum(capacity_weights)
        desired_sizes = [
            len(rows) * weight / capacity_total for weight in capacity_weights
        ]

        patient_items = list(patients.items())
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
                projected_size = len(client_rows[client_id]) + len(patient_rows)
                capacity_penalty = max(
                    0.0, projected_size / max(desired_sizes[client_id], 1.0) - 1.10
                )
                scores.append(
                    (
                        1.5 * class_score + 2.0 * size_score - 10.0 * capacity_penalty,
                        rng.random(),
                        client_id,
                    )
                )
            selected = max(scores)[2]
            client_rows[selected].extend(patient_rows)
            client_class_counts[selected].update(patient_counts)

        penalty, sizes, coverage, classes_per_client = _client_partition_penalty(
            client_rows, labels
        )
        target_error = sum(
            abs(client_class_counts[client_id][label] - desired_by_class[label][client_id])
            / max(total_by_class[label], 1)
            for label in labels
            for client_id in range(client_count)
        )
        size_error = sum(
            abs(sizes[client_id] - desired_sizes[client_id]) / max(len(rows), 1)
            for client_id in range(client_count)
        )
        candidate_score = penalty + target_error + size_error
        candidate = (
            candidate_score,
            [sorted(client, key=lambda row: row["img_id"]) for client in client_rows],
            sizes,
            coverage,
            classes_per_client,
        )
        if best is None or candidate_score < best[0]:
            best = candidate

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
    fieldnames, source_rows, label_mapping, median_age = preprocess_metadata(
        dataset_root, strict=strict
    )
    rows, duplicate_report = remove_exact_duplicates(dataset_root, source_rows)
    train_rows, test_rows = select_patient_split(
        rows, test_fraction=test_fraction, seed=seed
    )
    client_rows = partition_training_patients(
        train_rows, client_count=client_count, alpha=alpha, seed=seed
    )
    normalization = compute_channel_statistics(dataset_root, train_rows)
    verify_image_files(dataset_root, test_rows)

    processed_dir = dataset_root / "processed"
    write_csv(processed_dir / "PAD_metadata_preprocessed.csv", fieldnames, rows)
    write_csv(processed_dir / "PAD_metadata_train.csv", fieldnames, train_rows)
    write_csv(processed_dir / "PAD_metadata_test.csv", fieldnames, test_rows)
    # Prevent stale shards from an earlier run with a different client count
    # from being discovered by the runtime loader.
    for stale_shard in processed_dir.glob("client_*_train.csv"):
        stale_shard.unlink()
    for client_id, shard in enumerate(client_rows, start=1):
        write_csv(processed_dir / f"client_{client_id}_train.csv", fieldnames, shard)

    train_by_class = Counter(row["diagnostic"] for row in train_rows)
    test_by_class = Counter(row["diagnostic"] for row in test_rows)
    source_names = sorted({_image_source(row) for row in rows})
    summary = {
        "dataset": "PAD-UFES-20",
        "source": KAGGLE_HANDLE,
        "seed": seed,
        "test_fraction": test_fraction,
        "partition_alpha": alpha,
        "label_mapping": label_mapping,
        "median_age": median_age,
        "source_images": len(source_rows),
        "total_images": len(rows),
        "train_images": len(train_rows),
        "test_images": len(test_rows),
        "total_patients": len({row["patient_id"] for row in rows}),
        "train_patients": len({row["patient_id"] for row in train_rows}),
        "test_patients": len({row["patient_id"] for row in test_rows}),
        "train_class_counts": dict(train_by_class),
        "test_class_counts": dict(test_by_class),
        "actual_test_fraction": len(test_rows) / len(rows),
        "class_test_fractions": {
            diagnosis: test_by_class[diagnosis]
            / (train_by_class[diagnosis] + test_by_class[diagnosis])
            for diagnosis in sorted(label_mapping)
        },
        "train_source_counts": dict(
            Counter(_image_source(row) for row in train_rows)
        ),
        "test_source_counts": dict(
            Counter(_image_source(row) for row in test_rows)
        ),
        "class_source_counts": {
            diagnosis: {
                source_name: {
                    "train": sum(
                        row["diagnostic"] == diagnosis
                        and _image_source(row) == source_name
                        for row in train_rows
                    ),
                    "test": sum(
                        row["diagnostic"] == diagnosis
                        and _image_source(row) == source_name
                        for row in test_rows
                    ),
                }
                for source_name in source_names
            }
            for diagnosis in sorted(label_mapping)
        },
        "duplicate_audit": duplicate_report,
        "normalization": normalization,
        "verified_images": len(rows),
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
    print(
        f"  Images: {len(source_rows)} source, {len(rows)} clean, "
        f"{len(train_rows)} train, {len(test_rows)} test"
    )
    print(
        "  Exact-duplicate cleanup: "
        f"{duplicate_report['excluded_images']} images removed "
        f"({duplicate_report['conflicting_duplicate_groups']} conflicting groups)"
    )
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
