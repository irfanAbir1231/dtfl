import glob
import json
import logging
import math
import os
import re
from collections import Counter

import pandas as pd
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode


logger = logging.getLogger(__name__)

PAD_IMAGE_SIZE = 32
PAD_CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
PAD_CLASS_TO_LABEL = {name: index for index, name in enumerate(PAD_CLASS_NAMES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _is_prepared_pad_root(path):
    processed_dir = os.path.join(path, "processed")
    required = (
        "PAD_metadata_train.csv",
        "PAD_metadata_test.csv",
    )
    return all(os.path.isfile(os.path.join(processed_dir, name)) for name in required)


def resolve_pad_data_dir(data_dir):
    """Resolve either a PAD root or a parent directory containing ``PAD``."""
    candidates = []
    if data_dir:
        supplied = os.path.abspath(os.path.normpath(data_dir))
        candidates.extend((supplied, os.path.join(supplied, "PAD")))
    candidates.append(os.path.abspath(os.path.join("data", "PAD")))

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
        if _is_prepared_pad_root(candidate):
            return candidate

    searched = ", ".join(unique_candidates)
    raise FileNotFoundError(
        "PAD-UFES-20 has not been prepared. Expected a 'processed' directory "
        f"containing PAD_metadata_train.csv and PAD_metadata_test.csv. Searched: {searched}. "
        "Run 'python3 data/PAD/prepare_pad.py --download' from the repository root."
    )


def discover_pad_client_count(data_dir):
    dataset_root = resolve_pad_data_dir(data_dir)
    client_csvs = _discover_client_csvs(os.path.join(dataset_root, "processed"))
    return dataset_root, len(client_csvs)


def _discover_client_csvs(processed_dir):
    candidates = glob.glob(os.path.join(processed_dir, "client_*_train.csv"))
    indexed = []
    for path in candidates:
        match = re.fullmatch(r"client_(\d+)_train\.csv", os.path.basename(path))
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    client_csvs = [path for _, path in indexed]
    if not client_csvs:
        raise FileNotFoundError(
            f"No PAD client shard CSVs found in '{processed_dir}'."
        )
    identifiers = [identifier for identifier, _ in indexed]
    expected = list(range(1, len(indexed) + 1))
    if identifiers != expected:
        raise ValueError(
            f"PAD client shard IDs must be contiguous {expected}, found {identifiers}."
        )
    return client_csvs


def _load_pad_normalization(dataset_root):
    """Always return ImageNet normalization to match HAM10000's pipeline.

    PAD-specific statistics (computed at 64×64) introduced a domain shift
    relative to the ResNet model that was designed around ImageNet-normalised
    32×32 inputs.  Using the same normalisation as HAM10000 closes this gap.
    """
    return IMAGENET_MEAN, IMAGENET_STD


def _data_transforms_pad(strong_aug=False, mean=None, std=None):
    """Augmentation pipeline aligned with HAM10000 but stronger for PAD's
    extreme data scarcity (1,811 train images vs HAM's 7,991).

    Key differences from the old pipeline:
    - Image size 32×32 (matching HAM10000 and the ResNet architecture)
    - Aggressive geometric augmentation (rotation ±30°, affine translate/scale)
    - Stronger colour jitter to handle PAD's 3 different acquisition sources
    - GaussianBlur and RandomGrayscale for regularisation
    - RandomErasing for occlusion robustness
    """
    mean = IMAGENET_MEAN if mean is None else mean
    std = IMAGENET_STD if std is None else std
    fill = tuple(round(channel * 255) for channel in mean)

    # --- Train transforms ------------------------------------------------
    train_ops = [
        transforms.Resize(
            (PAD_IMAGE_SIZE, PAD_IMAGE_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(
            30, interpolation=InterpolationMode.BILINEAR, fill=fill
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.85, 1.15),
            interpolation=InterpolationMode.BILINEAR,
            fill=fill,
        ),
        transforms.RandomApply(
            [transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2, hue=0.04
            )],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)],
            p=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ]
    train_transform = transforms.Compose(train_ops)

    # --- Test transforms (deterministic, no augmentation) ----------------
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (PAD_IMAGE_SIZE, PAD_IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, test_transform


def _read_metadata(csv_path):
    dataframe = pd.read_csv(csv_path)
    required = {"patient_id", "lesion_id", "diagnostic", "img_id", "label", "path"}
    missing = required.difference(dataframe.columns)
    if missing:
        raise ValueError(
            f"PAD metadata file '{csv_path}' is missing columns: {sorted(missing)}"
        )
    if dataframe.empty:
        raise ValueError(f"PAD metadata file '{csv_path}' contains no rows.")
    for column in required:
        values = dataframe[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(
                f"PAD metadata file '{csv_path}' contains a blank '{column}' value."
            )
    if dataframe["img_id"].duplicated().any():
        duplicates = dataframe.loc[dataframe["img_id"].duplicated(), "img_id"].tolist()
        raise ValueError(
            f"PAD metadata file '{csv_path}' contains duplicate img_id values: {duplicates[:5]}"
        )
    return dataframe


def _balanced_sampler(dataframe, seed_key):
    """Aggressively rebalance diagnoses for PAD's extreme class imbalance.

    PAD has up to 16:1 class imbalance (BCC=663 vs MEL=42).  The old
    ``1/sqrt(count)`` weighting was too gentle — it only provided ~4:1
    rebalancing, causing the model to collapse minority classes.  Using
    ``1/count`` gives full inverse-frequency rebalancing so that each
    class is sampled with roughly equal probability per epoch.  Patient
    de-weighting is kept at ``1/sqrt`` to reduce repeated-patient bias
    without completely suppressing multi-lesion patients.
    """
    class_counts = Counter(dataframe["label"].astype(int))
    patient_counts = Counter(dataframe["patient_id"].astype(str))
    weights = [
        1.0
        / class_counts[int(row.label)]
        / math.sqrt(patient_counts[str(row.patient_id)])
        for row in dataframe.itertuples()
    ]
    generator = torch.Generator()
    generator.manual_seed(42 + sum(seed_key.encode("utf-8")))
    return data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


class PADDataset(data.Dataset):
    def __init__(self, dataframe, dataset_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.dataset_root = os.path.abspath(dataset_root)
        self.transform = transform
        self.target = self.dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = str(row["path"])
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.dataset_root, image_path)
        image_path = os.path.normpath(image_path)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"PAD image not found: '{image_path}'")

        with Image.open(image_path) as image:
            # Composite RGBA onto white background to preserve information
            # that would otherwise be lost (alpha=0 regions become black
            # with the default .convert("RGB")).
            if image.mode == "RGBA":
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[3])
                image = background
            else:
                image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, int(row["label"])


def get_dataloader_pad(
    data_dir,
    train_batch_size,
    test_batch_size,
    train_csv,
    test_csv,
    strong_aug=False,
    balanced_sampling=True,
):
    mean, std = _load_pad_normalization(data_dir)
    train_transform, test_transform = _data_transforms_pad(
        strong_aug=strong_aug, mean=mean, std=std
    )
    train_frame = _read_metadata(train_csv)
    train_dataset = PADDataset(train_frame, data_dir, transform=train_transform)
    test_dataset = PADDataset(
        _read_metadata(test_csv), data_dir, transform=test_transform
    )
    sampler = (
        _balanced_sampler(train_frame, os.path.basename(train_csv))
        if balanced_sampling
        else None
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=False,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, test_loader


def load_partition_data_pad(
    dataset,
    data_dir,
    partition_method,
    partition_alpha,
    client_number,
    batch_size,
    strong_aug=False,
    balanced_sampling=True,
):
    # PAD shards are created offline at patient level. The arguments remain in
    # the common loader signature for compatibility with main.py.
    del dataset, partition_method

    dataset_root = resolve_pad_data_dir(data_dir)
    processed_dir = os.path.join(dataset_root, "processed")
    client_csvs = _discover_client_csvs(processed_dir)
    if client_number != len(client_csvs):
        raise ValueError(
            f"Requested {client_number} PAD clients, but found {len(client_csvs)} "
            "prepared client shard CSVs. Re-run prepare_pad.py with the desired --clients value."
        )

    train_csv = os.path.join(processed_dir, "PAD_metadata_train.csv")
    test_csv = os.path.join(processed_dir, "PAD_metadata_test.csv")
    train_frame = _read_metadata(train_csv)
    test_frame = _read_metadata(test_csv)
    class_labels = sorted(train_frame["label"].astype(int).unique().tolist())
    expected_labels = list(range(len(PAD_CLASS_NAMES)))
    if class_labels != expected_labels:
        raise ValueError(
            f"PAD training labels must be {expected_labels}, found {class_labels}."
        )
    test_labels = sorted(test_frame["label"].astype(int).unique().tolist())
    if test_labels != expected_labels:
        raise ValueError(
            f"PAD test labels must be {expected_labels}, found {test_labels}."
        )
    for split_name, split_frame in (("training", train_frame), ("test", test_frame)):
        mapped_labels = split_frame["diagnostic"].map(PAD_CLASS_TO_LABEL)
        if mapped_labels.isna().any() or not mapped_labels.astype(int).equals(
            split_frame["label"].astype(int)
        ):
            raise ValueError(
                f"PAD diagnostic-to-label mapping is inconsistent in {split_name} metadata."
            )
    train_patients = set(train_frame["patient_id"].astype(str))
    test_patients = set(test_frame["patient_id"].astype(str))
    if train_patients & test_patients:
        raise ValueError("PAD patient leakage detected between train and test metadata.")
    if set(train_frame["img_id"].astype(str)) & set(test_frame["img_id"].astype(str)):
        raise ValueError("PAD image leakage detected between train and test metadata.")

    summary_path = os.path.join(processed_dir, "PAD_preprocessing_summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as summary_file:
            prepared_alpha = json.load(summary_file).get("partition_alpha")
        if prepared_alpha is not None and not math.isclose(
            float(prepared_alpha), float(partition_alpha), rel_tol=0.0, abs_tol=1e-12
        ):
            logger.warning(
                "PAD shards were prepared with alpha=%s, but runtime requested alpha=%s. "
                "Re-run prepare_pad.py to change the offline patient partition.",
                prepared_alpha,
                partition_alpha,
            )

    train_data_global, test_data_global = get_dataloader_pad(
        dataset_root,
        batch_size,
        batch_size,
        train_csv,
        test_csv,
        strong_aug=strong_aug,
        balanced_sampling=balanced_sampling,
    )
    train_data_num = 0
    test_data_num = len(test_frame)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}
    shard_image_ids = set()
    patient_owners = {}

    for client_idx, client_csv in enumerate(client_csvs):
        client_frame = _read_metadata(client_csv)
        client_image_ids = set(client_frame["img_id"].astype(str))
        overlap = shard_image_ids & client_image_ids
        if overlap:
            raise ValueError(
                f"PAD images occur in multiple client shards: {sorted(overlap)[:5]}"
            )
        shard_image_ids.update(client_image_ids)
        for patient_id in set(client_frame["patient_id"].astype(str)):
            if patient_id in patient_owners:
                raise ValueError(
                    f"PAD patient '{patient_id}' occurs in multiple client shards."
                )
            patient_owners[patient_id] = client_idx
        local_data_num = len(client_frame)
        train_data_num += local_data_num
        data_local_num_dict[client_idx] = local_data_num
        train_data_local, _ = get_dataloader_pad(
            dataset_root,
            batch_size,
            batch_size,
            client_csv,
            test_csv,
            strong_aug=strong_aug,
            balanced_sampling=balanced_sampling,
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_global
        logger.info(
            "PAD client_idx=%d, local_samples=%d, train_batches=%d, test_batches=%d",
            client_idx,
            local_data_num,
            len(train_data_local),
            len(test_data_global),
        )

    expected_train_ids = set(train_frame["img_id"].astype(str))
    if train_data_num != len(train_frame) or shard_image_ids != expected_train_ids:
        raise RuntimeError(
            "PAD client shards do not exactly cover the global train split: "
            f"{train_data_num} shard rows versus {len(train_frame)} global rows."
        )

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        len(PAD_CLASS_NAMES),
    )
