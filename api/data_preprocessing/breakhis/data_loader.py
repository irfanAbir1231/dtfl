"""
api/data_preprocessing/breakhis/data_loader.py
================================================
Federated data loader for the BreakHis Breast Cancer Histopathological Dataset.

Key design decisions (mirroring ISIC 2019 data_loader.py):
  - Uses kagglehub to auto-download
    ``waseemalastal/breakhis-breast-cancer-histopathological-dataset`` on first
    run.  Subsequent runs skip all preprocessing and go straight to DataLoader
    construction.
  - One-time preprocessing writes persistent train/test CSVs and per-client
    shard CSVs to ``{data_dir}/breakhis/``.
  - BreakHis filenames encode a patient ID
    (e.g. ``SOB_M_DC-14-11951-400-007.png`` → patient ``14-11951``).
    A **patient-level** GroupShuffleSplit 80/20 is used to prevent cross-slide
    leakage from the same patient appearing in both train and test splits.
    This mirrors the lesion_id grouping used in HAM10000.
  - Client shards are generated with Dirichlet-based (label-skewed)
    partitioning at the image level, matching the ``hetero`` partition_method
    used for CIFAR and ISIC 2019.
  - Class-weighted CrossEntropyLoss is handled by ``build_criterion()`` in
    main.py; this module exposes ``.target`` on every Dataset object so the
    weight computation works identically to HAM10000 and ISIC 2019.
  - All four magnification levels (40X, 100X, 200X, 400X) are pooled together
    as independent samples — the standard approach in most BreakHis papers.

8 tumour-subtype classes (multi-class classification):
  Benign:
    0 - A   (Adenosis)
    1 - F   (Fibroadenoma)
    2 - PT  (Phyllodes Tumor)
    3 - TA  (Tubular Adenoma)
  Malignant:
    4 - DC  (Ductal Carcinoma)
    5 - LC  (Lobular Carcinoma)
    6 - MC  (Mucinous Carcinoma)
    7 - PC  (Papillary Carcinoma)

Dataset directory structure (inside the kagglehub cache):
  BreaKHis_v1/
    histology_slides/
      breast/
        benign/
          SOB/
            adenosis/         → label 0
            fibroadenoma/     → label 1
            phyllodes_tumor/  → label 2
            tubular_adenoma/  → label 3
        malignant/
          SOB/
            ductal_carcinoma/         → label 4
            lobular_carcinoma/        → label 5
            mucinous_carcinoma/       → label 6
            papillary_carcinoma/      → label 7
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pandas as pd
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Constants ─────────────────────────────────────────────────────────────────

BREAKHIS_IMAGE_SIZE = 32   # same default as HAM10000 / ISIC 2019 in this framework

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Ordered class labels — the index is the integer label used throughout.
BREAKHIS_CLASSES = ["A", "F", "PT", "TA", "DC", "LC", "MC", "PC"]
CLASS_NUM = len(BREAKHIS_CLASSES)   # 8

# Maps lower-case folder names (as found in the Kaggle zip) → class index.
# We normalise names to lowercase + strip underscores/spaces for robustness.
_FOLDER_TO_CLASS_IDX: dict[str, int] = {
    # Benign subtypes
    "adenosis":          0,
    "fibroadenoma":      1,
    "phyllodes_tumor":   2,
    "phyllodestumor":    2,   # alt spelling seen in some versions
    "tubular_adenoma":   3,
    "tubularadenoma":    3,
    # Malignant subtypes
    "ductal_carcinoma":      4,
    "ductalcarcinoma":       4,
    "lobular_carcinoma":     5,
    "lobularcarcinoma":      5,
    "mucinous_carcinoma":    6,
    "mucinouscarcinoma":     6,
    "papillary_carcinoma":   7,
    "papillarycarcinoma":    7,
}

# Regex to parse the patient ID from a standard BreakHis filename.
# Format: SOB_{type}_{subtype}-{patient_id}-{magnification}-{seq}.{ext}
# Example: SOB_M_DC-14-11951-400-007.png  → patient = "14-11951"
_PATIENT_ID_RE = re.compile(
    r"^SOB_[BM]_[A-Za-z]+-(\d+-[\dA-Za-z]+)-\d+[Xx]?-\d+",
    re.IGNORECASE,
)


# ── Transforms ────────────────────────────────────────────────────────────────

def _data_transforms_breakhis(strong_aug: bool = False):
    """Return (train_transform, test_transform) for BreakHis.

    Augmentation rationale
    ----------------------
    Histopathology images have **no canonical orientation**, so we apply
    aggressive rotation (90° steps) and both flips.  Mild ``ColorJitter``
    compensates for inter-lab staining variation across the dataset's slides.

    When ``strong_aug=True`` the TITAN RandAugment + RandomErasing pipeline
    from ``utils.augmentation`` is used (same as HAM10000 / ISIC 2019 strong
    path).
    """
    if strong_aug:
        from utils.augmentation import build_titan_train_transform, build_titan_eval_transform
        train_transform = build_titan_train_transform(BREAKHIS_IMAGE_SIZE, strong=True)
        test_transform  = build_titan_eval_transform(BREAKHIS_IMAGE_SIZE)
        return train_transform, test_transform

    train_transform = transforms.Compose([
        transforms.Resize((BREAKHIS_IMAGE_SIZE, BREAKHIS_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.1,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((BREAKHIS_IMAGE_SIZE, BREAKHIS_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform


# ── Dataset class ─────────────────────────────────────────────────────────────

class BreakHisDataset(data.Dataset):
    """PyTorch Dataset for the BreakHis dataset (mirrors ISIC2019Dataset).

    Parameters
    ----------
    dataframe:
        A pandas DataFrame with at least ``path`` (absolute image path) and
        ``label`` (integer 0-7) columns.
    transform:
        torchvision transform applied to each image.
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        # Expose .target as a plain Python list so build_criterion() and
        # dataset_size computation in main.py work identically to HAM10000
        # and ISIC 2019.
        self.target = self.dataframe["label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = str(row["path"])
        label = int(row["label"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"BreakHis image not found: '{image_path}'")

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)

        return img, label


# ── Download & locate ─────────────────────────────────────────────────────────

def _download_breakhis() -> str:
    """Download the dataset via kagglehub and return the local root path.

    kagglehub caches the dataset on disk; repeated calls return the cached
    path without re-downloading.

    Returns
    -------
    str
        Path to the dataset root directory (the extracted zip root).
    """
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "kagglehub is required to download the BreakHis dataset. "
            "Install it with:  pip install kagglehub"
        ) from exc

    logger.info("Downloading BreakHis dataset via kagglehub …")
    dataset_root = kagglehub.dataset_download(
        "waseemalastal/breakhis-breast-cancer-histopathological-dataset"
    )
    logger.info("BreakHis dataset cached at: %s", dataset_root)
    return dataset_root


def _locate_image_roots(dataset_root: str) -> list[tuple[str, int]]:
    """Walk the extracted root and return (image_dir, label_index) pairs.

    The Kaggle version of the BreakHis dataset extracts to a tree like::

        <root>/
          BreaKHis_v1/
            histology_slides/
              breast/
                benign/
                  SOB/
                    adenosis/
                      40X/  100X/  200X/  400X/
                        SOB_B_A-14-22549AB-40-001.png  ...
                malignant/
                  SOB/
                    ductal_carcinoma/  ...

    We walk the whole tree looking for leaf directories that:
      1. Contain at least one image file (.png / .jpg / .tif / .bmp).
      2. Have a parent directory whose name matches a known subtype.

    Returns a list of (abs_dir_path, class_label) tuples.
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    results: list[tuple[str, int]] = []
    visited_subtypes: set[str] = set()

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        has_images = any(
            os.path.splitext(f)[1].lower() in image_extensions
            for f in filenames
        )
        if not has_images:
            continue

        # The subtype is encoded in one of the ancestor directory names.
        # Walk up the path components to find the first one that matches.
        parts = dirpath.replace("\\", "/").split("/")
        label_idx = None
        matched_subtype = None
        for part in reversed(parts):
            normalised = part.lower().replace(" ", "_")
            if normalised in _FOLDER_TO_CLASS_IDX:
                label_idx = _FOLDER_TO_CLASS_IDX[normalised]
                matched_subtype = normalised
                break

        if label_idx is None:
            logger.debug("Skipping directory (no known subtype found): %s", dirpath)
            continue

        results.append((dirpath, label_idx))
        visited_subtypes.add(matched_subtype)

    if not results:
        raise FileNotFoundError(
            f"Could not find any BreakHis image directories under: {dataset_root}\n"
            "Expected the standard BreakHis folder structure with subtype names "
            f"like: {list(_FOLDER_TO_CLASS_IDX.keys())}"
        )

    logger.info(
        "Located %d image directories covering subtypes: %s",
        len(results),
        sorted(visited_subtypes),
    )
    return results


# ── Master DataFrame ──────────────────────────────────────────────────────────

def _parse_patient_id(filename: str) -> str:
    """Extract the patient ID from a BreakHis filename.

    Returns the matched patient ID string, or the full stem if no match
    (graceful fallback so no rows are silently dropped).

    Examples
    --------
    >>> _parse_patient_id("SOB_M_DC-14-11951-400-007.png")
    '14-11951'
    >>> _parse_patient_id("SOB_B_A-14-22549AB-40-001.png")
    '14-22549AB'
    """
    stem = os.path.splitext(filename)[0]
    m = _PATIENT_ID_RE.match(stem)
    if m:
        return m.group(1)
    # Fall back: use the whole stem as a unique "patient" — prevents leakage
    # while still allowing the split to proceed.
    return stem


def _build_master_dataframe(image_roots: list[tuple[str, int]]) -> pd.DataFrame:
    """Enumerate all images in the located directories → DataFrame.

    Returns a DataFrame with columns:
      - ``path``       : absolute image path
      - ``label``      : integer class index 0-7
      - ``patient_id`` : patient identifier (for group-based splitting)
      - ``subtype``    : human-readable subtype name
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    records = []
    for img_dir, label_idx in image_roots:
        subtype_name = BREAKHIS_CLASSES[label_idx]
        for fname in os.listdir(img_dir):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in image_extensions:
                continue
            full_path = os.path.join(img_dir, fname)
            patient_id = _parse_patient_id(fname)
            records.append({
                "path":       full_path,
                "label":      label_idx,
                "patient_id": patient_id,
                "subtype":    subtype_name,
            })

    if not records:
        raise RuntimeError(
            "No image files were collected from the BreakHis dataset root. "
            "Check that the dataset was downloaded correctly."
        )

    df = pd.DataFrame(records)
    logger.info(
        "Master DataFrame: %d images | %d patients | %d classes",
        len(df),
        df["patient_id"].nunique(),
        df["label"].nunique(),
    )
    logger.info(
        "Class distribution:\n%s",
        df.groupby(["label", "subtype"]).size().to_string(),
    )
    return df


# ── Train/test split (patient-level) ─────────────────────────────────────────

def _patient_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level GroupShuffleSplit 80/20.

    Ensures that images from the same patient (multiple slides and
    magnification levels) cannot appear in both the train and test sets.

    Falls back to an image-level stratified split if GroupShuffleSplit is not
    available or if the number of unique patients is too small.
    """
    try:
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError:
        logger.warning(
            "scikit-learn not found; falling back to image-level stratified split."
        )
        return _stratified_train_test_split(df, test_size, seed)

    groups = df["patient_id"].values
    n_unique_patients = len(set(groups))

    if n_unique_patients < 5:
        logger.warning(
            "Only %d unique patients found; falling back to image-level stratified split.",
            n_unique_patients,
        )
        return _stratified_train_test_split(df, test_size, seed)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["label"], groups=groups))

    train_df = df.iloc[train_idx].copy()
    test_df  = df.iloc[test_idx].copy()

    # Log leakage check
    train_patients = set(train_df["patient_id"])
    test_patients  = set(test_df["patient_id"])
    overlap = train_patients.intersection(test_patients)
    if overlap:
        logger.warning(
            "Patient leakage detected: %d patients appear in both splits!",
            len(overlap),
        )
    else:
        logger.info(
            "Patient-level split: %d train images (%d patients) | "
            "%d test images (%d patients). No patient overlap. ✓",
            len(train_df), len(train_patients),
            len(test_df),  len(test_patients),
        )

    return train_df, test_df


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


# ── Preprocessing orchestrator ────────────────────────────────────────────────

def _ensure_preprocessed(data_dir: str) -> str:
    """Download and preprocess BreakHis if not already done.

    Returns the ``breakhis/`` working directory that holds all CSVs.
    """
    work_dir = os.path.normpath(os.path.join(data_dir, "breakhis"))
    os.makedirs(work_dir, exist_ok=True)

    train_csv = os.path.join(work_dir, "BreakHis_metadata_train.csv")
    test_csv  = os.path.join(work_dir, "BreakHis_metadata_test.csv")

    if os.path.exists(train_csv) and os.path.exists(test_csv):
        logger.info("Found existing BreakHis metadata CSVs — skipping preprocessing.")
        return work_dir

    # First run: download → locate → build → split → save
    dataset_root = _download_breakhis()
    image_roots  = _locate_image_roots(dataset_root)
    master_df    = _build_master_dataframe(image_roots)

    train_df, test_df = _patient_train_test_split(master_df, test_size=0.2, seed=42)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv,  index=False)

    logger.info("Train split: %d images saved to %s", len(train_df), train_csv)
    logger.info("Test  split: %d images saved to %s", len(test_df),  test_csv)

    return work_dir


# ── Dirichlet non-IID client partitioning ─────────────────────────────────────

def _dirichlet_partition(
    train_df: pd.DataFrame,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> dict:
    """Assign training indices to clients via Dirichlet label-skewed sampling.

    Mirrors the ``hetero`` partition used for ISIC 2019.

    Returns
    -------
    dict[int, list[int]]
        Mapping of client_id → list of row indices into ``train_df``.
    """
    rng = np.random.default_rng(seed)
    labels = train_df["label"].values
    num_classes = CLASS_NUM
    n_total = len(train_df)

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
    (1-indexed to match the ISIC 2019 / HAM10000 convention).

    Returns a sorted list of absolute paths to the client CSV files.
    """
    client_csvs = sorted([
        os.path.join(work_dir, f"client_{i + 1}_train.csv")
        for i in range(num_clients)
    ])

    if all(os.path.exists(p) for p in client_csvs):
        logger.info(
            "Found %d existing BreakHis client shard CSVs — skipping partitioning.",
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

def get_dataloader_breakhis(
    train_csv: str,
    test_csv: str,
    train_bs: int,
    test_bs: int,
    strong_aug: bool = False,
):
    """Build a (train_loader, test_loader) pair from CSV paths."""
    train_transform, test_transform = _data_transforms_breakhis(strong_aug=strong_aug)

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    train_ds = BreakHisDataset(train_df, transform=train_transform)
    test_ds  = BreakHisDataset(test_df,  transform=test_transform)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, shuffle=True,  drop_last=False,
        num_workers=0, pin_memory=False,
    )
    test_dl = data.DataLoader(
        dataset=test_ds,  batch_size=test_bs,  shuffle=False, drop_last=False,
        num_workers=0, pin_memory=False,
    )
    return train_dl, test_dl


# ── Public API ─────────────────────────────────────────────────────────────────

def load_partition_data_breakhis(
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
      1. Downloads the BreakHis dataset via kagglehub (if not cached).
      2. Parses and splits into train/test CSVs (saved in
         ``data_dir/breakhis/``), using a patient-level GroupShuffleSplit.
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
    train_csv = os.path.join(work_dir, "BreakHis_metadata_train.csv")
    test_csv  = os.path.join(work_dir, "BreakHis_metadata_test.csv")

    # 2. Ensure client shards exist
    # When partition_method == "homo" use a very large alpha to approximate IID.
    alpha = partition_alpha if partition_method != "homo" else 1000.0
    client_csvs = _ensure_client_csvs(
        work_dir, train_csv, client_number, alpha, seed=42
    )

    # 3. Build global loaders
    train_data_global, test_data_global = get_dataloader_breakhis(
        train_csv, test_csv, batch_size, batch_size, strong_aug=strong_aug
    )

    global_train_df = pd.read_csv(train_csv)
    global_test_df  = pd.read_csv(test_csv)

    test_data_num  = len(global_test_df)
    train_data_num = 0
    class_num      = CLASS_NUM

    logging.info("BreakHis train_dl_global batches = %d", len(train_data_global))
    logging.info("BreakHis test_dl_global  batches = %d", len(test_data_global))

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
            "BreakHis client_idx=%d, local_sample_number=%d",
            client_idx, local_data_num,
        )

        train_dl_local, _ = get_dataloader_breakhis(
            client_csv, test_csv, batch_size, batch_size, strong_aug=strong_aug
        )
        logging.info(
            "BreakHis client_idx=%d, batch_num_train_local=%d",
            client_idx, len(train_dl_local),
        )

        train_data_local_dict[client_idx] = train_dl_local
        # All clients share the same global test set (mirrors HAM10000 / ISIC 2019)
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
