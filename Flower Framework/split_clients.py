import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path


def _dirichlet(size, alpha, rng):
    """Draw a symmetric Dirichlet sample without requiring NumPy."""
    draws = [rng.gammavariate(alpha, 1.0) for _ in range(size)]
    total = sum(draws)
    return [value / total for value in draws]


def _cap_proportions(proportions, maximum_share):
    """Project proportions onto a simplex with an upper bound per client."""
    capped = list(proportions)
    for _ in range(100):
        over = [i for i, value in enumerate(capped) if value > maximum_share]
        if not over:
            break

        excess = sum(capped[i] - maximum_share for i in over)
        for i in over:
            capped[i] = maximum_share

        eligible = [
            i for i, value in enumerate(capped)
            if value < maximum_share - 1e-12
        ]
        eligible_total = sum(capped[i] for i in eligible)
        if eligible_total <= 0:
            addition = excess / len(eligible)
            for i in eligible:
                capped[i] += addition
        else:
            for i in eligible:
                capped[i] += excess * capped[i] / eligible_total

    return capped


def _read_lesion_groups(train_csv_path):
    with train_csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {train_csv_path}")

        required = {"lesion_id", "label", "image_id"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{train_csv_path} is missing required columns: {sorted(missing)}"
            )

        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    lesion_groups = {}
    for row in rows:
        lesion_id = row["lesion_id"]
        label = row["label"]
        if lesion_id not in lesion_groups:
            lesion_groups[lesion_id] = {"label": label, "rows": []}
        elif lesion_groups[lesion_id]["label"] != label:
            raise ValueError(
                f"Lesion {lesion_id!r} has inconsistent labels "
                f"{lesion_groups[lesion_id]['label']!r} and {label!r}."
            )
        lesion_groups[lesion_id]["rows"].append(row)

    return fieldnames, rows, lesion_groups


def _build_candidate(
    lesion_groups,
    num_clients,
    alpha,
    minimum_clients_per_class,
    maximum_class_share,
    seed,
):
    rng = random.Random(seed)
    lesions_by_class = defaultdict(list)
    total_images = 0
    for lesion_id, group in lesion_groups.items():
        image_count = len(group["rows"])
        lesions_by_class[group["label"]].append((lesion_id, image_count))
        total_images += image_count

    # Quantity targets are deliberately bounded. Label distributions remain
    # non-IID, but no client is intentionally assigned a near-zero capacity.
    capacity_multipliers = [
        rng.uniform(0.72, 1.28) for _ in range(num_clients)
    ]
    multiplier_sum = sum(capacity_multipliers)
    capacity_targets = [
        total_images * multiplier / multiplier_sum
        for multiplier in capacity_multipliers
    ]

    client_lesions = [[] for _ in range(num_clients)]
    client_image_counts = [0 for _ in range(num_clients)]
    class_image_counts = {
        label: [0 for _ in range(num_clients)]
        for label in lesions_by_class
    }

    for label, original_lesions in lesions_by_class.items():
        lesions = list(original_lesions)
        rng.shuffle(lesions)
        lesions.sort(key=lambda item: item[1], reverse=True)

        class_total = sum(image_count for _, image_count in lesions)
        proportions = _cap_proportions(
            _dirichlet(num_clients, alpha, rng),
            maximum_class_share,
        )
        class_targets = [
            proportion * class_total for proportion in proportions
        ]

        # Seed the strongest Dirichlet destinations so every class reaches the
        # requested minimum client coverage before assigning other lesions.
        seeded_clients = sorted(
            range(num_clients),
            key=lambda client_id: class_targets[client_id],
            reverse=True,
        )[:minimum_clients_per_class]
        for client_id in seeded_clients:
            lesion_id, image_count = lesions.pop()
            client_lesions[client_id].append(lesion_id)
            client_image_counts[client_id] += image_count
            class_image_counts[label][client_id] += image_count

        maximum_images_for_class = math.floor(
            maximum_class_share * class_total
        )
        for lesion_id, image_count in lesions:
            valid_clients = [
                client_id
                for client_id in range(num_clients)
                if (
                    class_image_counts[label][client_id] + image_count
                    <= maximum_images_for_class
                )
            ]
            if not valid_clients:
                valid_clients = list(range(num_clients))

            def assignment_score(client_id):
                class_deficit = (
                    class_targets[client_id]
                    - class_image_counts[label][client_id]
                ) / max(class_targets[client_id], 1.0)
                capacity_deficit = (
                    capacity_targets[client_id]
                    - client_image_counts[client_id]
                ) / max(capacity_targets[client_id], 1.0)
                return (
                    1.6 * class_deficit
                    + capacity_deficit
                    + rng.random() * 1e-9
                )

            selected_client = max(valid_clients, key=assignment_score)
            client_lesions[selected_client].append(lesion_id)
            client_image_counts[selected_client] += image_count
            class_image_counts[label][selected_client] += image_count

    return client_lesions, client_image_counts, class_image_counts


def _constraint_failures(
    client_image_counts,
    class_image_counts,
    minimum_images,
    maximum_size_ratio,
    minimum_classes,
    minimum_clients_per_class,
    maximum_class_share,
):
    failures = []
    if min(client_image_counts) < minimum_images:
        failures.append(
            f"minimum client size {min(client_image_counts)} < {minimum_images}"
        )

    size_ratio = max(client_image_counts) / min(client_image_counts)
    if size_ratio > maximum_size_ratio:
        failures.append(
            f"maximum/minimum size ratio {size_ratio:.3f} > "
            f"{maximum_size_ratio:.3f}"
        )

    classes_per_client = [
        sum(counts[client_id] > 0 for counts in class_image_counts.values())
        for client_id in range(len(client_image_counts))
    ]
    if min(classes_per_client) < minimum_classes:
        failures.append(
            f"minimum classes on a client {min(classes_per_client)} < "
            f"{minimum_classes}"
        )

    clients_per_class = {
        label: sum(count > 0 for count in counts)
        for label, counts in class_image_counts.items()
    }
    if min(clients_per_class.values()) < minimum_clients_per_class:
        failures.append(
            f"minimum clients containing a class "
            f"{min(clients_per_class.values())} < {minimum_clients_per_class}"
        )

    ownership = {
        label: max(counts) / sum(counts)
        for label, counts in class_image_counts.items()
    }
    if max(ownership.values()) > maximum_class_share + 1e-12:
        failures.append(
            f"maximum one-client class share {max(ownership.values()):.3f} > "
            f"{maximum_class_share:.3f}"
        )

    return failures


def _write_and_verify(
    train_csv_path,
    fieldnames,
    original_rows,
    lesion_groups,
    client_lesions,
    client_image_counts,
    class_image_counts,
):
    output_dir = train_csv_path.parent
    assigned_lesions = [
        lesion_id
        for lesions in client_lesions
        for lesion_id in lesions
    ]
    if len(assigned_lesions) != len(set(assigned_lesions)):
        raise RuntimeError("A lesion was assigned to more than one client.")
    if set(assigned_lesions) != set(lesion_groups):
        raise RuntimeError("The client partitions do not cover every lesion.")

    written_image_ids = []
    for client_id, lesion_ids in enumerate(client_lesions):
        client_rows = []
        for lesion_id in lesion_ids:
            client_rows.extend(lesion_groups[lesion_id]["rows"])
        client_rows.sort(key=lambda row: row["image_id"])
        written_image_ids.extend(row["image_id"] for row in client_rows)

        client_path = output_dir / f"client_{client_id + 1}_train.csv"
        with client_path.open("w", newline="") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(client_rows)

    original_image_ids = [row["image_id"] for row in original_rows]
    if len(written_image_ids) != len(set(written_image_ids)):
        raise RuntimeError("An image was written to more than one client CSV.")
    if set(written_image_ids) != set(original_image_ids):
        raise RuntimeError("The generated client CSVs do not cover all images.")

    print("\n--- Constrained non-IID client distribution ---")
    labels = sorted(
        class_image_counts,
        key=lambda value: int(value) if value.isdigit() else value,
    )
    for client_id, image_count in enumerate(client_image_counts):
        counts = {
            label: class_image_counts[label][client_id]
            for label in labels
            if class_image_counts[label][client_id] > 0
        }
        print(
            f"Client {client_id + 1}: {image_count:4d} images | "
            f"labels {counts}"
        )

    ratio = max(client_image_counts) / min(client_image_counts)
    print(f"\nMaximum/minimum client-size ratio: {ratio:.3f}")
    for label in labels:
        counts = class_image_counts[label]
        owners = sum(count > 0 for count in counts)
        maximum_share = max(counts) / sum(counts)
        print(
            f"Class {label}: {owners} clients | "
            f"maximum one-client share {maximum_share:.3f}"
        )
    print(
        f"Verified {len(original_rows)} unique training images and "
        f"{len(lesion_groups)} unique lesions across 8 client CSVs."
    )


def create_non_iid_clients(
    train_csv_path,
    num_clients=8,
    alpha=0.5,
    minimum_images=500,
    maximum_size_ratio=4.0,
    minimum_classes=3,
    minimum_clients_per_class=3,
    maximum_class_share=0.5,
    seed=42,
    maximum_attempts=10_000,
):
    """Create constrained label-skewed, lesion-level non-IID client shards."""
    train_csv_path = Path(train_csv_path)
    fieldnames, rows, lesion_groups = _read_lesion_groups(train_csv_path)

    last_failures = []
    for attempt in range(maximum_attempts):
        candidate_seed = seed + attempt
        (
            client_lesions,
            client_image_counts,
            class_image_counts,
        ) = _build_candidate(
            lesion_groups,
            num_clients,
            alpha,
            minimum_clients_per_class,
            maximum_class_share,
            candidate_seed,
        )
        last_failures = _constraint_failures(
            client_image_counts,
            class_image_counts,
            minimum_images,
            maximum_size_ratio,
            minimum_classes,
            minimum_clients_per_class,
            maximum_class_share,
        )
        if not last_failures:
            print(
                f"Accepted deterministic partition seed {candidate_seed} "
                f"(base seed {seed}, attempt {attempt + 1})."
            )
            _write_and_verify(
                train_csv_path,
                fieldnames,
                rows,
                lesion_groups,
                client_lesions,
                client_image_counts,
                class_image_counts,
            )
            return

    raise RuntimeError(
        f"No valid partition found after {maximum_attempts} attempts. "
        f"Last failures: {last_failures}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate constrained lesion-level non-IID HAM10000 clients."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "Dataset/HAM10000_metadata_train.csv"
        ),
    )
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--minimum-images", type=int, default=500)
    parser.add_argument("--maximum-size-ratio", type=float, default=4.0)
    parser.add_argument("--minimum-classes", type=int, default=3)
    parser.add_argument("--minimum-clients-per-class", type=int, default=3)
    parser.add_argument("--maximum-class-share", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    create_non_iid_clients(
        train_csv_path=args.train_csv,
        num_clients=args.clients,
        alpha=args.alpha,
        minimum_images=args.minimum_images,
        maximum_size_ratio=args.maximum_size_ratio,
        minimum_classes=args.minimum_classes,
        minimum_clients_per_class=args.minimum_clients_per_class,
        maximum_class_share=args.maximum_class_share,
        seed=args.seed,
    )
