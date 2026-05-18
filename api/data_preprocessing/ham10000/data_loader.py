import glob
import logging
import os

import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

HAM10000_IMAGE_SIZE = 32
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _resolve_dataset_root(data_dir):
    dataset_root = os.path.abspath(data_dir)
    dataset_dir = os.path.join(dataset_root, "Dataset")
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"HAM10000 dataset directory not found. Expected '{dataset_dir}'."
        )
    return dataset_root, dataset_dir


def _discover_client_csvs(dataset_dir):
    client_csvs = sorted(glob.glob(os.path.join(dataset_dir, "client_*_train.csv")))
    if not client_csvs:
        raise FileNotFoundError(
            f"No client shard CSVs were found in '{dataset_dir}'."
        )
    return client_csvs


def _data_transforms_ham10000():
    train_transform = transforms.Compose([
        transforms.Resize((HAM10000_IMAGE_SIZE, HAM10000_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=359),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((HAM10000_IMAGE_SIZE, HAM10000_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


class HAM10000Dataset(data.Dataset):
    def __init__(self, dataframe, dataset_root, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.dataset_root = os.path.abspath(dataset_root)
        self.transform = transform
        self.target = self.dataframe["label"].astype(int).tolist()

    def __len__(self):
        return len(self.dataframe)

    def _resolve_image_path(self, relative_path):
        image_path = relative_path
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.dataset_root, image_path)
        image_path = os.path.normpath(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"HAM10000 image not found: '{image_path}'")
        return image_path

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self._resolve_image_path(row["path"])
        label = int(row["label"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)

        return image, label


def _read_metadata(csv_path):
    dataframe = pd.read_csv(csv_path)
    required_columns = {"label", "path"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata file '{csv_path}' is missing required columns: {sorted(missing_columns)}"
        )
    return dataframe


def get_dataloader_ham10000(data_dir, train_bs, test_bs, csv_path_train, csv_path_test):
    dataset_root, _ = _resolve_dataset_root(data_dir)
    train_transform, test_transform = _data_transforms_ham10000()

    train_df = _read_metadata(csv_path_train)
    test_df = _read_metadata(csv_path_test)

    train_ds = HAM10000Dataset(train_df, dataset_root=dataset_root, transform=train_transform)
    test_ds = HAM10000Dataset(test_df, dataset_root=dataset_root, transform=test_transform)

    class_counts = train_df['label'].value_counts().to_dict()
    sample_weights = [1.0 / class_counts[label] for label in train_ds.target]
    sampler = data.WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, sampler=sampler, drop_last=False)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl


def load_partition_data_ham10000(dataset, data_dir, partition_method, partition_alpha, client_number, batch_size):
    del dataset, partition_method, partition_alpha

    dataset_root, dataset_dir = _resolve_dataset_root(data_dir)
    client_csvs = _discover_client_csvs(dataset_dir)

    if client_number > len(client_csvs):
        raise ValueError(
            f"Requested {client_number} clients, but only found {len(client_csvs)} client shard CSVs."
        )

    if client_number < len(client_csvs):
        logger.warning(
            "Using the first %s HAM10000 client shards out of %s available shards.",
            client_number,
            len(client_csvs),
        )
        client_csvs = client_csvs[:client_number]

    global_train_csv = os.path.join(dataset_dir, "HAM10000_metadata_train.csv")
    global_test_csv = os.path.join(dataset_dir, "HAM10000_metadata_test.csv")
    global_train_df = _read_metadata(global_train_csv)
    global_test_df = _read_metadata(global_test_csv)

    class_num = int(global_train_df["label"].nunique())
    train_data_num = 0
    test_data_num = len(global_test_df)

    train_data_global, test_data_global = get_dataloader_ham10000(
        dataset_root,
        batch_size,
        batch_size,
        global_train_csv,
        global_test_csv,
    )
    logging.info("train_dl_global number = %s", len(train_data_global))
    logging.info("test_dl_global number = %s", len(test_data_global))

    data_local_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for client_idx, client_csv in enumerate(client_csvs):
        client_df = _read_metadata(client_csv)
        local_data_num = len(client_df)
        train_data_num += local_data_num
        data_local_num_dict[client_idx] = local_data_num
        logging.info("client_idx = %d, local_sample_number = %d", client_idx, local_data_num)

        train_data_local, _ = get_dataloader_ham10000(
            dataset_root,
            batch_size,
            batch_size,
            client_csv,
            global_test_csv,
        )
        logging.info(
            "client_idx = %d, batch_num_train_local = %d, batch_num_test_local = %d",
            client_idx,
            len(train_data_local),
            len(test_data_global),
        )
        train_data_local_dict[client_idx] = train_data_local
        test_data_local_dict[client_idx] = test_data_global

    return (
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
