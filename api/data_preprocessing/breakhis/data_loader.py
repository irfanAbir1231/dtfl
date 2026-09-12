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
    (e.g. ``SOB_M_DC-14-11951-400-007.png`` → patient ``14-11951``) and the
    magnification level (40X, 100X, 200X, 400X).
    A **patient-level** GroupShuffleSplit 80/20 is used to prevent cross-slide
    leakage from the same patient appearing in both train and test splits.
  - Client shards are generated with Dirichlet-based (label-skewed)
    partitioning at the image level, matching the ``hetero`` partition_method
    used for CIFAR and ISIC 2019.
  - Class-weighted CrossEntropyLoss is handled by ``build_criterion()`` in
    main.py; this module exposes ``.target`` on every Dataset object so the
    weight computation works identically to HAM10000 and ISIC 2019.
  - All four magnification levels (40X, 100X, 200X, 400X) are pooled together
    as independent samples — the standard approach in most BreakHis papers.
  - On first run, pixel mean/std are computed from the training images and
    saved to ``{work_dir}/breakhis_norm_stats.json`` so that BreakHis-specific
    (not ImageNet) normalization is used.  This is critical because H&E
    histopathology has a very different colour distribution from ImageNet.

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

import json
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

BREAKHIS_IMAGE_SIZE = 64   # BreakHis native: 700×460 pixels.
                           # 64×64 gives 4× more pixels than 32×32 (major accuracy
                           # gain) while fitting in 24 GB GPU at batch_size=100
                           # with 10 simultaneous client+server model copies.
                           # 128×128 caused CUDA OOM: feature maps at the split
                           # layer (B×16×128×128) exceed available VRAM.

# Realistic BreakHis H&E pixel stats (pinkish-purple background/nuclei)
_FALLBACK_MEAN = [0.78, 0.60, 0.72]
_FALLBACK_STD  = [0.15, 0.18, 0.14]


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
# Example: SOB_M_DC-14-11951-400-007.png  → patient = "14-11951", mag = "400"
_PATIENT_ID_RE = re.compile(
    r"^SOB_[BM]_[A-Za-z]+-(\d+-[\dA-Za-z]+)-(\d+)[Xx]?-\d+",
    re.IGNORECASE,
)


# ── Dataset-specific normalization stats ─────────────────────────────────────

def _load_or_compute_norm_stats(
    work_dir: str,
    train_df: pd.DataFrame | None = None,
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Load saved norm stats or compute them from training images.

    H&E histopathology images have a fundamentally different colour distribution
    from ImageNet (pinkish-purple instead of natural-scene colours).  Using the
    wrong statistics shifts every normalised pixel by ~0.3 in the R and B
    channels, crippling the model's ability to learn colour-based features.

    This function computes the actual per-channel pixel mean and std from a
    sample of training images and caches the result in
    ``{work_dir}/breakhis_norm_stats.json`` so subsequent runs are fast.

    Parameters
    ----------
    work_dir:
        Directory where ``breakhis_norm_stats.json`` will be written.
    train_df:
        DataFrame with a ``path`` column.  Required on the first call (when no
        cached stats exist).  Pass ``None`` on subsequent calls to just load.
    n_samples:
        Number of training images to sample for statistics computation.
    seed:
        Random seed for reproducible sampling.

    Returns
    -------
    (mean, std) each a list of 3 floats in [0, 1] range.
    """
    stats_path = os.path.join(work_dir, "breakhis_norm_stats.json")

    if os.path.exists(stats_path):
        with open(stats_path) as f:
            d = json.load(f)
        logger.info(
            "Loaded BreakHis norm stats from cache: mean=%s  std=%s",
            [round(v, 4) for v in d["mean"]],
            [round(v, 4) for v in d["std"]],
        )
        return d["mean"], d["std"]

    if train_df is None:
        logger.warning(
            "No cached norm stats and no train_df provided — falling back to ImageNet stats."
        )
        return _FALLBACK_MEAN, _FALLBACK_STD

    logger.info(
        "Computing BreakHis pixel statistics from %d training images …", n_samples
    )
    sample_df = train_df.sample(
        min(n_samples, len(train_df)), random_state=seed
    ).reset_index(drop=True)

    pixel_sum    = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels     = 0
    errors       = 0

    for path in sample_df["path"]:
        try:
            with Image.open(path) as img:
                arr = np.array(img.convert("RGB")).astype(np.float64) / 255.0
                pixel_sum    += arr.sum(axis=(0, 1))
                pixel_sq_sum += (arr ** 2).sum(axis=(0, 1))
                n_pixels     += arr.shape[0] * arr.shape[1]
        except Exception as exc:
            logger.debug("Skipping image %s: %s", path, exc)
            errors += 1

    if errors:
        logger.warning("Skipped %d images during norm-stat computation.", errors)

    if n_pixels == 0:
        logger.warning("No pixels processed — falling back to ImageNet stats.")
        return _FALLBACK_MEAN, _FALLBACK_STD

    mean = (pixel_sum / n_pixels).tolist()
    var  = pixel_sq_sum / n_pixels - (pixel_sum / n_pixels) ** 2
    std  = np.maximum(var, 1e-8) ** 0.5
    std  = std.tolist()

    logger.info(
        "BreakHis norm stats computed: mean=%s  std=%s",
        [round(v, 4) for v in mean],
        [round(v, 4) for v in std],
    )

    with open(stats_path, "w") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)

    return mean, std


# ── Transforms ────────────────────────────────────────────────────────────────

class _IdentityTransform:
    """No-op transform (picklable replacement for ``lambda img: img``).

    ``transforms.Lambda(lambda img: img)`` cannot be pickled by Python's
    multiprocessing module, which is required when ``num_workers > 0`` in a
    DataLoader.  This named class is identical in behaviour but fully picklable.
    """
    def __call__(self, img):
        return img

    def __repr__(self) -> str:
        return "IdentityTransform()"

def _data_transforms_breakhis(
    norm_mean: list[float] | None = None,
    norm_std:  list[float] | None = None,
    strong_aug: bool = False,
):
    """Return (train_transform, test_transform) for BreakHis.

    Augmentation rationale (overfitting regime: train~58%, test~42%)
    -----------------------------------------------
    The model is **overfitting** (train >> test gap = 16 pp).  The root cause
    in federated learning is that each client's Dirichlet-skewed local shard
    has severe class imbalance, causing per-client overfitting.  This is
    addressed both here (regularising augmentation) and in the DataLoader
    (WeightedRandomSampler, see ``_make_weighted_sampler``).

    Augmentation choices:

    1. **Full 4-way discrete rotation** (0°/90°/180°/270°):
       Tissue has no canonical orientation.  Using all four 90° steps gives
       complete rotational invariance and 4× effective data augmentation.

    2. **RandomResizedCrop (scale 0.80–1.0)**:
       Slightly wider range than before; simulates different viewing regions
       on the slide and reduces position-specific memorisation.

    3. **ColorJitter (moderate)**:
       Compensates for inter-lab H&E staining variation.

    4. **GaussianBlur (p=0.15)**:
       Mimics focus variation in whole-slide imaging.

    5. **RandomErasing (p=0.2)** — RE-ENABLED for overfitting regime:
       Randomly masks out 2–20% of the image.  Acts as a strong regulariser
       that forces the model to classify from partial views, preventing it
       from latching onto a single discriminative region.

    6. **BreakHis-specific normalization** (not ImageNet):
       H&E images have R≈0.78, G≈0.60, B≈0.72.  Using ImageNet stats
       ([0.485, 0.456, 0.406]) shifts all pixels by up to 0.30 in the wrong
       direction.  Actual dataset stats are computed once at preprocessing time.

    Parameters
    ----------
    norm_mean, norm_std:
        Per-channel mean and std in [0, 1] computed from actual BreakHis
        training images.  Fall back to ImageNet stats if None.
    strong_aug:
        When True, uses the TITAN RandAugment + RandomErasing pipeline from
        ``utils.augmentation``.
    """
    mean = norm_mean if norm_mean is not None else _FALLBACK_MEAN
    std  = norm_std  if norm_std  is not None else _FALLBACK_STD

    if strong_aug:
        from utils.augmentation import build_titan_train_transform, build_titan_eval_transform
        train_transform = build_titan_train_transform(BREAKHIS_IMAGE_SIZE, strong=True)
        test_transform  = build_titan_eval_transform(BREAKHIS_IMAGE_SIZE)
        return train_transform, test_transform

    train_transform = transforms.Compose([
        # ── High-resolution isotropic crop directly from raw image ───────────
        # Crops a 1:1 square patch directly from the raw 700×460 slide and
        # resizes it to BREAKHIS_IMAGE_SIZE (64×64). Preserves cellular
        # morphology, round nuclei, and fine chromatin texture without any
        # anisotropic squashing (no aspect ratio distortion).
        transforms.RandomResizedCrop(
            BREAKHIS_IMAGE_SIZE,
            scale=(0.75, 1.0),
            ratio=(0.95, 1.05),
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        # ── Dihedral D4 geometry (histopathology tissue has no canonical orientation)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        # Full discrete 4-step rotation: 0° / 90° / 180° / 270°.
        # Histopathology tissue is rotationally symmetric — a tumour cell looks
        # identical at every 90° step. Using all four steps gives complete
        # rotational invariance and effectively 4× the training variety.
        # NOTE: _IdentityTransform is used instead of lambda for picklability
        # (DataLoader with num_workers>0 requires all transforms to be picklable).
        transforms.RandomChoice([
            _IdentityTransform(),                            # 0°  (no-op)
            transforms.RandomRotation(degrees=(90,  90)),   # 90°
            transforms.RandomRotation(degrees=(180, 180)),  # 180°
            transforms.RandomRotation(degrees=(270, 270)),  # 270°
        ]),
        # ── Colour (mild — accounts for H&E staining variations without blurring chromatin)
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.03,
        ),
        # ── To tensor + BreakHis-specific normalisation ──────────────────
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # ── Gentle RandomErasing (mild regularisation without destroying tiny diagnostic clusters)
        transforms.RandomErasing(p=0.10, scale=(0.02, 0.10)),
    ])

    test_transform = transforms.Compose([
        # Preserves 1.52 aspect ratio during downsampling (shorter edge 460 -> 64, width -> 97)
        transforms.Resize(BREAKHIS_IMAGE_SIZE),
        # Crops the central 64×64 biopsy region (removes slide borders, 1:1 isotropic)
        transforms.CenterCrop(BREAKHIS_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
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


def _parse_magnification(filename: str) -> str:
    """Extract the magnification level (40, 100, 200, 400) from a filename.

    Returns the magnification string, or "unknown" on no match.

    Examples
    --------
    >>> _parse_magnification("SOB_M_DC-14-11951-400-007.png")
    '400'
    >>> _parse_magnification("SOB_B_A-14-22549AB-40-001.png")
    '40'
    """
    stem = os.path.splitext(filename)[0]
    m = _PATIENT_ID_RE.match(stem)
    if m:
        return m.group(2)
    return "unknown"


def _build_master_dataframe(image_roots: list[tuple[str, int]]) -> pd.DataFrame:
    """Enumerate all images in the located directories → DataFrame.

    Returns a DataFrame with columns:
      - ``path``         : absolute image path
      - ``label``        : integer class index 0-7
      - ``patient_id``   : patient identifier (for group-based splitting)
      - ``subtype``      : human-readable subtype name
      - ``magnification``: magnification level ("40", "100", "200", "400")
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
            patient_id    = _parse_patient_id(fname)
            magnification = _parse_magnification(fname)
            records.append({
                "path":          full_path,
                "label":         label_idx,
                "patient_id":    patient_id,
                "subtype":       subtype_name,
                "magnification": magnification,
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
    logger.info(
        "Magnification distribution:\n%s",
        df["magnification"].value_counts().sort_index().to_string(),
    )
    return df


SPLIT_VERSION = "v3_stratified_patient"


# ── Train/test split (patient-level stratified) ──────────────────────────────

def _patient_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class patient-level stratified 80/20 train/test split.

    Guarantees:
      1. Every single tumour subtype class (0 to 7) is represented in BOTH train
         and test splits. (Unstratified GroupShuffleSplit randomly left out
         Fibroadenoma and Papillary Carcinoma completely from the test set).
      2. No patient's slides overlap between train and test (0% patient-level data leakage).
      3. Test set contains a balanced ~20% of patients for each class.
      4. Fully deterministic and pure Python/NumPy (no dependency on scikit-learn).
    """
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for label in sorted(df["label"].unique()):
        class_df = df[df["label"] == label]
        patient_ids = np.array(sorted(class_df["patient_id"].unique()))
        rng.shuffle(patient_ids)

        n_patients = len(patient_ids)
        if n_patients <= 1:
            # Single patient for a class: split images 80/20 to guarantee test representation
            logger.warning(
                "Class %d has only 1 patient (%s); splitting images 80/20.",
                label, patient_ids[0],
            )
            class_indices = class_df.index.to_numpy()
            rng.shuffle(class_indices)
            n_test = max(1, int(round(len(class_indices) * test_size)))
            test_indices.extend(class_indices[:n_test])
            train_indices.extend(class_indices[n_test:])
            continue

        n_test_patients = max(1, int(round(n_patients * test_size)))
        # Ensure at least 1 patient in train
        if n_patients - n_test_patients < 1:
            n_test_patients = n_patients - 1

        test_pats = set(patient_ids[:n_test_patients])
        train_pats = set(patient_ids[n_test_patients:])

        test_indices.extend(class_df[class_df["patient_id"].isin(test_pats)].index)
        train_indices.extend(class_df[class_df["patient_id"].isin(train_pats)].index)

    train_df = df.loc[train_indices].reset_index(drop=True)
    test_df  = df.loc[test_indices].reset_index(drop=True)

    # Verification checks
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
            "Stratified patient split: %d train images (%d patients) | "
            "%d test images (%d patients). No patient overlap. ✓",
            len(train_df), len(train_patients),
            len(test_df),  len(test_patients),
        )

    logger.info(
        "Train class distribution: %s",
        train_df["label"].value_counts().sort_index().to_dict(),
    )
    logger.info(
        "Test  class distribution: %s",
        test_df["label"].value_counts().sort_index().to_dict(),
    )

    return train_df, test_df


# ── Preprocessing orchestrator ────────────────────────────────────────────────

def _ensure_preprocessed(data_dir: str) -> str:
    """Download and preprocess BreakHis if not already done.

    Includes automatic validation of the split version and class completeness.
    If existing CSVs were created with an older/unstratified split (e.g. missing
    classes in the test set), it automatically purges the stale files and
    regenerates the clean stratified split.

    Returns the ``breakhis/`` working directory that holds all CSVs.
    """
    work_dir = os.path.normpath(os.path.join(data_dir, "breakhis"))
    os.makedirs(work_dir, exist_ok=True)

    train_csv = os.path.join(work_dir, "BreakHis_metadata_train.csv")
    test_csv  = os.path.join(work_dir, "BreakHis_metadata_test.csv")
    version_file = os.path.join(work_dir, ".split_version")

    needs_rebuild = False
    if not (os.path.exists(train_csv) and os.path.exists(test_csv)):
        needs_rebuild = True
    elif not os.path.exists(version_file):
        logger.info("Outdated BreakHis split detected (no version marker). Triggering rebuild...")
        needs_rebuild = True
    else:
        try:
            with open(version_file) as f:
                v = f.read().strip()
            if v != SPLIT_VERSION:
                logger.info("Split version mismatch ('%s' != '%s'). Triggering rebuild...", v, SPLIT_VERSION)
                needs_rebuild = True
            else:
                # Integrity check: test set must contain all 8 classes
                test_df_check = pd.read_csv(test_csv)
                if test_df_check["label"].nunique() < CLASS_NUM:
                    logger.warning(
                        "Existing BreakHis test split is corrupted (only %d of %d classes present). "
                        "Forcing rebuild...",
                        test_df_check["label"].nunique(), CLASS_NUM,
                    )
                    needs_rebuild = True
        except Exception as exc:
            logger.warning("Error validating existing split (%s). Triggering rebuild...", exc)
            needs_rebuild = True

    if not needs_rebuild:
        logger.info("Found valid BreakHis metadata CSVs (%s) — skipping preprocessing.", SPLIT_VERSION)
        # Still try to compute norm stats if missing (handles upgrade from old runs)
        _load_or_compute_norm_stats(
            work_dir,
            train_df=pd.read_csv(train_csv) if not os.path.exists(
                os.path.join(work_dir, "breakhis_norm_stats.json")
            ) else None,
        )
        return work_dir

    # Purge any old client CSVs before rebuilding to avoid partition mismatch
    for fname in os.listdir(work_dir):
        if fname.startswith("client_") and fname.endswith("_train.csv"):
            try:
                os.remove(os.path.join(work_dir, fname))
            except OSError:
                pass

    # First run or rebuild: download → locate → build → split → save → stats
    dataset_root = _download_breakhis()
    image_roots  = _locate_image_roots(dataset_root)
    master_df    = _build_master_dataframe(image_roots)

    train_df, test_df = _patient_train_test_split(master_df, test_size=0.2, seed=42)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv,  index=False)

    try:
        with open(version_file, "w") as f:
            f.write(SPLIT_VERSION)
    except OSError as exc:
        logger.warning("Could not write version file %s: %s", version_file, exc)

    logger.info("Train split: %d images saved to %s", len(train_df), train_csv)
    logger.info("Test  split: %d images saved to %s", len(test_df),  test_csv)

    # Compute and cache dataset-specific normalization stats from training images.
    _load_or_compute_norm_stats(work_dir, train_df=train_df, n_samples=1000)

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


# ── Weighted sampler ────────────────────────────────────────────────────────────────

def _make_weighted_sampler(dataset: BreakHisDataset) -> data.WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances class frequencies.

    Motivation
    ----------
    In federated learning, each client receives a Dirichlet-skewed shard
    (alpha=0.5 by default).  This creates severe within-client class imbalance:
    one or two classes dominate each client's local data.

    Without correction the model learns to predict the dominant local class
    and ignores rare classes, causing high training accuracy (for the dominant
    class) but poor generalisation on the balanced test set.

    WeightedRandomSampler assigns each sample a weight inversely proportional
    to its class frequency.  As a result, every epoch sees each class roughly
    equally often, preventing the model from over-specialising to the dominant
    class in each client's shard.

    This is applied ONLY to training DataLoaders (per-client and global train).
    The test DataLoader keeps the natural distribution.

    Parameters
    ----------
    dataset:
        A ``BreakHisDataset`` instance.  Uses ``dataset.target`` (the label
        list) to compute per-class counts.

    Returns
    -------
    WeightedRandomSampler with ``num_samples = len(dataset)`` and
    ``replacement=True`` (required for oversampling minority classes).
    """
    import torch
    labels       = np.array(dataset.target, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=CLASS_NUM).astype(np.float64)
    class_counts  = np.maximum(class_counts, 1.0)   # avoid division by zero

    # Weight for each sample = 1 / (count of its class) — minority classes
    # get higher weight so they are sampled more often.
    sample_weights = 1.0 / class_counts[labels]

    logger.info(
        "WeightedRandomSampler class counts: %s",
        {i: int(c) for i, c in enumerate(class_counts.astype(int))},
    )
    return data.WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(dataset),
        replacement=True,
    )


# ── DataLoader factories ───────────────────────────────────────────────────────

def get_dataloader_breakhis(
    train_csv: str,
    test_csv: str,
    train_bs: int,
    test_bs: int,
    strong_aug: bool = False,
    norm_mean: list[float] | None = None,
    norm_std:  list[float] | None = None,
    use_weighted_sampler: bool = False,
):
    """Build a (train_loader, test_loader) pair from CSV paths.

    Parameters
    ----------
    norm_mean, norm_std:
        Dataset-specific pixel normalization stats.  Computed from the actual
        BreakHis training images by ``_load_or_compute_norm_stats`` and passed
        here so every DataLoader (global and per-client) uses identical stats.
    use_weighted_sampler:
        When True, replaces ``shuffle=True`` with a ``WeightedRandomSampler``
        that balances class frequencies per epoch.  Use for training loaders
        to counteract the class imbalance introduced by Dirichlet partitioning.
    """
    train_transform, test_transform = _data_transforms_breakhis(
        norm_mean=norm_mean, norm_std=norm_std, strong_aug=strong_aug
    )

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    train_ds = BreakHisDataset(train_df, transform=train_transform)
    test_ds  = BreakHisDataset(test_df,  transform=test_transform)

    # Weighted sampler balances class frequencies in each training epoch.
    # shuffle=True is mutually exclusive with a custom sampler in PyTorch.
    if use_weighted_sampler:
        sampler = _make_weighted_sampler(train_ds)
        train_dl = data.DataLoader(
            dataset=train_ds, batch_size=train_bs,
            sampler=sampler, drop_last=False,
            num_workers=2, pin_memory=True,
        )
    else:
        train_dl = data.DataLoader(
            dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False,
            num_workers=2, pin_memory=True,
        )

    test_dl = data.DataLoader(
        dataset=test_ds,  batch_size=test_bs,  shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True,
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
      3. Computes BreakHis-specific pixel normalization stats and caches them.
      4. Creates per-client shard CSVs using Dirichlet partitioning.

    Subsequent calls skip steps 1-4 and load directly from the saved CSVs.

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

    # 2. Load dataset-specific normalization stats (computed from actual images).
    #    Every DataLoader — global and per-client — uses the SAME stats so that
    #    the normalisation is consistent across the entire federated system.
    norm_mean, norm_std = _load_or_compute_norm_stats(work_dir)

    # 3. Ensure client shards exist
    # When partition_method == "homo" use a very large alpha to approximate IID.
    alpha = partition_alpha if partition_method != "homo" else 1000.0
    client_csvs = _ensure_client_csvs(
        work_dir, train_csv, client_number, alpha, seed=42
    )

    # 4. Build global loaders (shuffle=True, no sample repetition)
    train_data_global, test_data_global = get_dataloader_breakhis(
        train_csv, test_csv, batch_size, batch_size,
        strong_aug=strong_aug, norm_mean=norm_mean, norm_std=norm_std,
        use_weighted_sampler=False,
    )

    global_train_df = pd.read_csv(train_csv)
    global_test_df  = pd.read_csv(test_csv)

    test_data_num  = len(global_test_df)
    train_data_num = 0
    class_num      = CLASS_NUM

    logging.info("BreakHis train_dl_global batches = %d", len(train_data_global))
    logging.info("BreakHis test_dl_global  batches = %d", len(test_data_global))

    # 5. Build per-client local loaders with shuffle=True.
    #    Each sample is visited once per local epoch, preventing duplicate memorization.
    #    Class imbalance is handled smoothly by build_criterion() in main.py without
    #    penalizing the dominant class (Ductal Carcinoma, 44% of test data).
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
            client_csv, test_csv, batch_size, batch_size,
            strong_aug=strong_aug, norm_mean=norm_mean, norm_std=norm_std,
            use_weighted_sampler=False,
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
