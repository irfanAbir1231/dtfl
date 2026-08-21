"""
api/data_preprocessing/isic2019/data_loader.py
===============================================
Federated data loader for the ISIC 2019 Challenge Dataset.

Key design decisions (mirroring HAM10000 data_loader.py):
  - Uses kagglehub to auto-download ``andrewmvd/isic-2019`` on first run.
  - One-time preprocessing writes persistent train/test CSVs and per-client
    shard CSVs to ``{data_dir}/isic2019/``.  Subsequent runs skip all
    preprocessing and go straight to DataLoader construction.
  - ISIC 2019 has no ``lesion_id`` grouping concept, so a stratified 80/20
    split is used instead of GroupShuffleSplit.
  - Client shards are generated with Dirichlet-based (label-skewed) partitioning
    at the image level, matching the ``hetero`` partition_method used for CIFAR.
  - Class-weighted CrossEntropyLoss is handled by ``build_criterion()`` in
    main.py; this module exposes ``.target`` on every Dataset object so the
    weight computation works identically to HAM10000.

9 diagnostic classes (matching the one-hot columns in the ground-truth CSV):
  0 - MEL   (Melanoma)
  1 - NV    (Melanocytic nevus)
  2 - BCC   (Basal cell carcinoma)
  3 - AK    (Actinic keratosis)
  4 - BKL   (Benign keratosis)
  5 - DF    (Dermatofibroma)
  6 - VASC  (Vascular lesion)
  7 - SCC   (Squamous cell carcinoma)
  8 - UNK   (None of the above)
"""

from __future__ import annotations

import logging
import os
import csv

import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────────

ISIC2019_IMAGE_SIZE = 32

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Ordered list of one-hot column names in the ground-truth CSV → label index.
ISIC2019_CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
CLASS_NUM = len(ISIC2019_CLASSES)   # 9


# ── Transforms ─────────────────────────────────────────────────────────────────

def _data_transforms_isic2019(strong_aug: bool = False):
    """Return (train_transform, test_transform) for ISIC 2019.

    Strategy rationale
    ------------------
    Compared with HAM10000 we add two extra training augmentations:

    * ``RandomRotation(20)`` — dermoscopic images have no canonical
      orientation; rotation invariance significantly boosts generalisation.
    * ``ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)``
      — ISIC 2019 aggregates images from three acquisition sites (BCN, HAM,
      MSK) with different lighting and colour calibration, so colour jitter
      encourages the network to focus on structural features rather than
      site-specific colour artefacts.

    When ``strong_aug=True`` the TITAN RandAugment + RandomErasing pipeline
    from ``utils.augmentation`` is used (same as HAM10000 strong path).
    """
    if strong_aug:
        from utils.augmentation import build_titan_train_transform, build_titan_eval_transform
        train_transform = build_titan_train_transform(ISIC2019_IMAGE_SIZE, strong=True)
        test_transform  = build_titan_eval_transform(ISIC2019_IMAGE_SIZE)
        return train_transform, test_transform

    train_transform = transforms.Compose([
        transforms.Resize((ISIC2019_IMAGE_SIZE, ISIC2019_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((ISIC2019_IMAGE_SIZE, ISIC2019_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


# ── Dataset class ─────────────────────────────────────────────────────────────

class ISIC2019Dataset(data.Dataset):
    """PyTorch Dataset for the ISIC 2019 challenge (mirrors HAM10000Dataset).

    Parameters
    ----------
    dataframe:
        A pandas DataFrame with at least ``path`` (absolute image path) and
        ``label`` (integer 0-8) columns.
    transform:
        torchvision transform applied to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        # Expose .target as a plain Python list so build_criterion() and
        # dataset_size computation in main.py work identically to HAM10000.
        self.target = self.dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = str(row["path"])
        label = int(row["label"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"ISIC2019 image not found: '{image_path}'")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label


# ── Download & preprocessing ──────────────────────────────────────────────────

def _download_isic2019() -> str:
    """Download the dataset via kagglehub and return the local root path.

    kagglehub caches the dataset on disk; repeated calls return the cached
    path without re-downloading.

    Returns
    -------
    str
        Path to the dataset root directory (contains ``train/`` images and the
        ground-truth CSV).
    """
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the ISIC 2019 dataset. "
            "Install it with:  pip install kagglehub"
        ) from exc

    logger.info("Downloading ISIC 2019 dataset via kagglehub …")
    dataset_root = kagglehub.dataset_download("andrewmvd/isic-2019")
    logger.info("ISIC 2019 dataset cached at: %s", dataset_root)
    return dataset_root


def _locate_dataset_files(dataset_root: str):
    """Walk the downloaded root to find the ground-truth CSV and image dir.

    The kagglehub version of ``andrewmvd/isic-2019`` has this layout::

        <root>/
          ISIC_2019_Training_GroundTruth.csv
          ISIC_2019_Training_Input/
            ISIC_0024306.jpg
            ...

    Returns
    -------
    gt_csv_path : str
        Absolute path to the ground-truth CSV.
    image_dir : str
        Absolute path to the directory containing ``*.jpg`` images.
    """
    gt_csv = None
    image_dir = None

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        for fname in filenames:
            if fname == "ISIC_2019_Training_GroundTruth.csv" and gt_csv is None:
                gt_csv = os.path.join(dirpath, fname)
        for dname in dirnames:
            # Accept any sub-directory whose name contains "Input" or "train"
            if ("Input" in dname or "train" in dname.lower()) and image_dir is None:
                candidate = os.path.join(dirpath, dname)
                # Check it actually contains at least one jpg
                if any(f.endswith(".jpg") for f in os.listdir(candidate)):
                    image_dir = candidate

    if gt_csv is None:
        raise FileNotFoundError(
            f"Could not locate 'ISIC_2019_Training_GroundTruth.csv' under: {dataset_root}"
        )
    if image_dir is None:
        raise FileNotFoundError(
            f"Could not locate the ISIC 2019 training image directory under: {dataset_root}"
        )

    return gt_csv, image_dir


def _build_master_dataframe(gt_csv_path: str, image_dir: str) -> pd.DataFrame:
    """Parse the one-hot ground-truth CSV → integer labels + image paths.

    The CSV has columns: ``image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK``
    Each row has exactly one column set to 1.0; the rest are 0.0.

    Returns a DataFrame with columns: ``image_id``, ``label``, ``path``.
    """
    logger.info("Parsing ground-truth CSV: %s", gt_csv_path)
    df = pd.read_csv(gt_csv_path)

    # Robustness: strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Identify which ISIC classes are present in this CSV version
    present_classes = [c for c in ISIC2019_CLASSES if c in df.columns]
    if not present_classes:
        raise ValueError(
            f"Ground-truth CSV has none of the expected class columns "
            f"({ISIC2019_CLASSES}). Found columns: {list(df.columns)}"
        )

    # Derive integer label from one-hot encoding
    label_matrix = df[present_classes].values  # (N, C)
    labels = label_matrix.argmax(axis=1)        # index within present_classes

    # If CSV is missing some classes (e.g. no UNK column) remap to global idx
    global_indices = [ISIC2019_CLASSES.index(c) for c in present_classes]
    labels = np.array([global_indices[l] for l in labels], dtype=np.int64)

    # Image column is typically named "image" in the Kaggle CSV
    image_col = "image" if "image" in df.columns else df.columns[0]
    image_ids = df[image_col].str.strip().tolist()

    # Map image IDs → absolute paths (.jpg)
    paths = []
    missing = 0
    for img_id in image_ids:
        p = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.exists(p):
            paths.append(None)
            missing += 1
        else:
            paths.append(p)

    if missing > 0:
        logger.warning(
            "%d / %d image files were not found in '%s'. "
            "These rows will be dropped.",
            missing, len(image_ids), image_dir,
        )

    result = pd.DataFrame({
        "image_id": image_ids,
        "label": labels,
        "path": paths,
    })
    result = result.dropna(subset=["path"]).reset_index(drop=True)
    logger.info(
        "Master DataFrame: %d images, %d classes",
        len(result), int(result["label"].nunique()),
    )
    return result


def _stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 train/test split preserving class proportions."""
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(sss.split(df, df["label"]))
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def _ensure_preprocessed(data_dir: str) -> str:
    """Download and preprocess ISIC 2019 if not already done.

    Returns the ``isic2019/`` working directory that holds all CSVs.
    """
    work_dir = os.path.normpath(os.path.join(data_dir, "isic2019"))
    os.makedirs(work_dir, exist_ok=True)

    train_csv = os.path.join(work_dir, "ISIC2019_metadata_train.csv")
    test_csv  = os.path.join(work_dir, "ISIC2019_metadata_test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        logger.info("Found existing ISIC 2019 metadata CSVs — skipping preprocessing.")
        return work_dir

    # First run: download → parse → split → save
    dataset_root = _download_isic2019()
    gt_csv_path, image_dir = _locate_dataset_files(dataset_root)
    master_df = _build_master_dataframe(gt_csv_path, image_dir)

    train_df, test_df = _stratified_train_test_split(master_df, test_size=0.2, seed=42)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    logger.info("Train split: %d images saved to %s", len(train_df), train_csv)
    logger.info("Test  split: %d images saved to %s", len(test_df),  test_csv)

    # Quick leakage sanity check (image-level — no lesion grouping in ISIC 2019)
    overlap = set(train_df["image_id"]).intersection(set(test_df["image_id"]))
    if overlap:
        logger.warning("Data leakage detected: %d images appear in both splits!", len(overlap))
    else:
        logger.info("No image overlap between train and test splits. ✓")

    return work_dir


# ── Dirichlet non-IID client partitioning ────────────────────────────────────

def _dirichlet_partition(
    train_df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> dict:
    """Assign training indices to clients via Dirichlet label-skewed sampling.

    Mirrors the ``hetero`` partition used for CIFAR-10 in data_loader.py.
    Each class's indices are split across clients via a Dirichlet draw,
    ensuring a controllable degree of non-IID distribution.

    Returns
    -------
    dict[int, list[int]]
        Mapping of client_id → list of row indices into ``train_df``.
    """
    rng = np.random.default_rng(seed)
    labels = train_df["label"].values
    num_classes = CLASS_NUM
    n_total = len(train_df)

    # Retry until every client gets at least 10 samples
    min_size = 0
    net_dataidx_map = {}
    while min_size < 10:
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(labels == k)[0]
            rng.shuffle(idx_k)
            if len(idx_k) == 0:
                continue
            proportions = rng.dirichlet(np.repeat(alpha, num_clients))
            # Balance: don't give to clients that are already full
            proportions = np.array([
                p * (len(idx_j) < n_total / num_clients)
                for p, idx_j in zip(proportions, idx_batch)
            ])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
            ]
            min_size = min(len(idx_j) for idx_j in idx_batch)

    for j in range(num_clients):
        rng.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return net_dataidx_map


def _ensure_client_csvs(
    work_dir: str,
    train_csv: str,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> list[str]:
    """Create per-client shard CSVs if they don't already exist.

    CSV files are named ``client_1_train.csv`` … ``client_N_train.csv``
    (1-indexed to match the HAM10000 convention read by ``split_clients.py``).

    Returns a sorted list of absolute paths to the client CSV files.
    """
    client_csvs = sorted([
        os.path.join(work_dir, f"client_{i + 1}_train.csv")
        for i in range(num_clients)
    ])

    if all(os.path.exists(p) for p in client_csvs):
        logger.info(
            "Found %d existing ISIC 2019 client shard CSVs — skipping partitioning.",
            num_clients,
        )
        return client_csvs

    logger.info(
        "Creating %d non-IID client shards (Dirichlet alpha=%.2f) …",
        num_clients, alpha,
    )
    train_df = pd.read_csv(train_csv)
    net_dataidx_map = _dirichlet_partition(train_df, num_clients, alpha, seed)

    for client_id, indices in net_dataidx_map.items():
        client_df = train_df.iloc[indices].copy()
        client_df.to_csv(client_csvs[client_id], index=False)
        logger.info(
            "Client %d: %d images | label dist: %s",
            client_id + 1,
            len(client_df),
            client_df["label"].value_counts().sort_index().to_dict(),
        )

    return client_csvs


# ── DataLoader factories ───────────────────────────────────────────────────────

def get_dataloader_isic2019(
    train_csv: str,
    test_csv: str,
    train_bs: int,
    test_bs: int,
    strong_aug: bool = False,
):
    """Build a (train_loader, test_loader) pair from CSV paths."""
    train_transform, test_transform = _data_transforms_isic2019(strong_aug=strong_aug)

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    train_ds = ISIC2019Dataset(train_df, transform=train_transform)
    test_ds  = ISIC2019Dataset(test_df,  transform=test_transform)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True,  drop_last=False,
        num_workers=0, pin_memory=False,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,  batch_size=test_bs,  shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    return train_dl, test_dl


# ── Public API ────────────────────────────────────────────────────────────────

def load_partition_data_isic2019(
    dataset,
    data_dir: str,
    partition_method: str,
    partition_alpha: float,
    client_number: int,
    batch_size: int,
    strong_aug: bool = False,
):
    """Top-level loader matching the signature of all other partition loaders.

    On the first call this function:
      1. Downloads the ISIC 2019 dataset via kagglehub (if not cached).
      2. Parses and splits into train/test CSVs (saved in ``data_dir/isic2019/``).
      3. Creates per-client shard CSVs using Dirichlet partitioning.

    Subsequent calls skip steps 1-3 and load directly from the saved CSVs.

    Returns
    -------
    tuple of length 8:
        (train_data_num, test_data_num,
         train_data_global, test_data_global,
         data_local_num_dict, train_data_local_dict, test_data_local_dict,
         class_num)
    """
    del dataset  # unused; kept for API compatibility

    # 1. Ensure raw data is downloaded and preprocessed
    work_dir  = _ensure_preprocessed(data_dir)
    train_csv = os.path.join(work_dir, "ISIC2019_metadata_train.csv")
    test_csv  = os.path.join(work_dir, "ISIC2019_metadata_test.csv")

    # 2. Ensure client shards exist
    # When partition_method == "hetero" use the provided alpha;
    # for "homo" use a very large alpha to approximate IID.
    alpha = partition_alpha if partition_method != "homo" else 1000.0
    client_csvs = _ensure_client_csvs(
        work_dir, train_csv, client_number, alpha, seed=42
    )

    # 3. Build global loaders
    train_data_global, test_data_global = get_dataloader_isic2019(
        train_csv, test_csv, batch_size, batch_size, strong_aug=strong_aug
    )

    global_test_df  = pd.read_csv(test_csv)
    global_train_df = pd.read_csv(train_csv)

    test_data_num  = len(global_test_df)
    train_data_num = 0
    class_num      = CLASS_NUM

    logging.info("ISIC2019 train_dl_global batches = %d", len(train_data_global))
    logging.info("ISIC2019 test_dl_global  batches = %d", len(test_data_global))

    # 4. Build per-client local loaders
    data_local_num_dict   = {}
    train_data_local_dict = {}
    test_data_local_dict  = {}

    for client_idx, client_csv in enumerate(client_csvs):
        client_df      = pd.read_csv(client_csv)
        local_data_num = len(client_df)
        train_data_num += local_data_num
        data_local_num_dict[client_idx] = local_data_num

        logging.info(
            "ISIC2019 client_idx=%d, local_sample_number=%d",
            client_idx, local_data_num,
        )

        train_dl_local, _ = get_dataloader_isic2019(
            client_csv, test_csv, batch_size, batch_size, strong_aug=strong_aug
        )
        logging.info(
            "ISIC2019 client_idx=%d, batch_num_train_local=%d",
            client_idx, len(train_dl_local),
        )

        train_data_local_dict[client_idx] = train_dl_local
        # All clients share the same global test set (mirrors HAM10000)
        test_data_local_dict[client_idx]  = test_data_global

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
