import glob
import logging
import os

import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


logger = logging.getLogger(__name__)

PAD_IMAGE_SIZE = 32
PAD_CLASS_NAMES = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
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
    client_csvs = sorted(
        glob.glob(os.path.join(dataset_root, "processed", "client_*_train.csv"))
    )
    if not client_csvs:
        raise FileNotFoundError(
            f"No PAD client shard CSVs found in '{os.path.join(dataset_root, 'processed')}'."
        )
    return dataset_root, len(client_csvs)


def _data_transforms_pad(strong_aug=False):
    if strong_aug:
        from utils.augmentation import (
            build_titan_eval_transform,
            build_titan_train_transform,
        )

        train_transform = build_titan_train_transform(PAD_IMAGE_SIZE, strong=True)
        test_transform = build_titan_eval_transform(PAD_IMAGE_SIZE)
    else:
        # Match the active HAM10000 image pipeline for a controlled comparison.
        train_transform = transforms.Compose(
            [
                transforms.Resize((PAD_IMAGE_SIZE, PAD_IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((PAD_IMAGE_SIZE, PAD_IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
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
    return dataframe


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
):
    train_transform, test_transform = _data_transforms_pad(strong_aug=strong_aug)
    train_dataset = PADDataset(
        _read_metadata(train_csv), data_dir, transform=train_transform
    )
    test_dataset = PADDataset(
        _read_metadata(test_csv), data_dir, transform=test_transform
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
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
):
    # PAD shards are created offline at patient level. The arguments remain in
    # the common loader signature for compatibility with main.py.
    del dataset, partition_method, partition_alpha

    dataset_root = resolve_pad_data_dir(data_dir)
    processed_dir = os.path.join(dataset_root, "processed")
    client_csvs = sorted(glob.glob(os.path.join(processed_dir, "client_*_train.csv")))
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

    train_data_global, test_data_global = get_dataloader_pad(
        dataset_root,
        batch_size,
        batch_size,
        train_csv,
        test_csv,
        strong_aug=strong_aug,
    )
    train_data_num = 0
    test_data_num = len(test_frame)
    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx, client_csv in enumerate(client_csvs):
        client_frame = _read_metadata(client_csv)
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

    if train_data_num != len(train_frame):
        raise RuntimeError(
            f"PAD client shards contain {train_data_num} rows, but the global train split "
            f"contains {len(train_frame)} rows."
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
