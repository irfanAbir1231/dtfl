# HAM10000 Dataset Documentation & Analysis

This document provides a highly detailed analysis of the dataset located in the `Dataset/` directory. Based on the files present, this structure corresponds to the established **HAM10000 (Human Against Machine with 10000 training images)** dataset, which is a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

---

## 1. Directory Structure

The dataset contains the following files and directories:

*   **Metadata Information**:
    *   `HAM10000_metadata.csv`: The main file linking demographics, diagnosis, and image IDs.
*   **Raw Image Folders**:
    *   `HAM10000_images_part_1/`: Directory containing part 1 of the high-resolution JPEG images.
    *   `HAM10000_images_part_2/`: Directory containing part 2 of the high-resolution JPEG images.
    *(Note: Images had to be split into two parts due to upload constraints, but they form a single unified collection of 10,015 images).*
*   **Pre-processed Image Data (MNIST-like Replacements)**:
    *   `hmnist_28_28_RGB.csv`: Flattened arrays of 28x28 pixel images in RGB color (3 channels).
    *   `hmnist_28_28_L.csv`: Flattened arrays of 28x28 pixel images in Grayscale (Luminance, 1 channel).
    *   `hmnist_8_8_RGB.csv`: Flattened 8x8 pixel color images.
    *   `hmnist_8_8_L.csv`: Flattened 8x8 pixel grayscale images.

---

## 2. Deep Dive: `HAM10000_metadata.csv`

The `HAM10000_metadata.csv` file is the master key for this dataset. It contains exactly **10,015 rows** (each representing a single image) and the following **7 columns**:

### Column Descriptions & Data Types
1.  **`lesion_id`** *(String)*: A unique identifier for the actual physical skin lesion. 
2.  **`image_id`** *(String)*: A unique identifier for the photograph (the actual image file is named `<image_id>.jpg` in the images folders).
3.  **`dx`** *(String / Categorical)*: The target variable (Diagnosis). See the "Diagnoses" section below.
4.  **`dx_type`** *(String / Categorical)*: How the diagnosis was confirmed (e.g., `histo` for histopathology, `follow_up` for clinical follow-up, `consensus` for expert consensus, `confocal` for in-vivo confocal microscopy).
5.  **`age`** *(Numeric)*: The age of the patient in years (rounded to intervals of 5). Contains a very small percentage of missing values (NaN).
6.  **`sex`** *(String / Categorical)*: The sex of the patient (`male`, `female`, or `unknown`).
7.  **`localization`** *(String / Categorical)*: The specific body part where the lesion was found (e.g., `back`, `lower extremity`, `trunk`, `face`, `abdomen`).

### Duplicates & Relationships (CRITICAL)
*   **Exact Duplicate Rows Check**: Are there completely duplicate image files/rows? **No (0 exact duplicates)**. Every `image_id` is unique. 
*   **Lesion vs Image Relationship**: Multiple photographs can belong to a single physical skin lesion. Therefore, multiple **`image_id`**s map to exactly the same **`lesion_id`**.
*   **Class Distribution & Lesion Breakdown**: Out of the 10,015 total images, there are actually only 7,470 unique lesions. The breakdown per class is as follows:

| Class (`dx`) | Total Images | Unique Lesions (`lesion_id`) | Duplicate Images (Same Lesion) |
| :--- | :--- | :--- | :--- |
| **nv** | 6,705 | 5,403 | 1,302 |
| **mel** | 1,113 | 614 | 499 |
| **bkl** | 1,099 | 727 | 372 |
| **bcc** | 514 | 327 | 187 |
| **akiec** | 327 | 228 | 99 |
| **vasc** | 142 | 98 | 44 |
| **df** | 115 | 73 | 42 |
| **TOTAL** | **10,015** | **7,470** | **2,545** |

*   **Data Leakage Warning:** Because multiple images belong to the exact same lesion (as shown in the far-right column above), doing a simple random split for Training and Validation/Test sets can result in **staggering data leakage**. If Image A and Image B represent the exact same lesion, and Image A is in the training set while Image B is in the testing set, the model will cheat by recognizing the specific lesion (e.g. hair structure, lighting, angles) rather than learning general disease traits.
*   **Best Practice:** Always group by `lesion_id` when performing train/validation splits (e.g., using `GroupKFold` or `GroupShuffleSplit` in scikit-learn) so that all 2,545 duplicate representations fall directly within the exact same split as their primary partner.

---

## 3. Diagnoses / Target Classes (`dx`)

The dataset contains 7 distinct classes of skin lesions. This is structured as a **highly imbalanced multi-class classification** problem. The 7 classes are:

1.  **nv** (Melanocytic nevi): Benign neoplasms of melanocytes. This is the overwhelming majority class (approx. 6,700 images).
2.  **mel** (Melanoma): A malignant neoplasm derived from melanocytes. The most dangerous form of skin cancer.
3.  **bkl** (Benign keratosis-like lesions): Includes solar lentigines / seborrheic keratoses and lichen-planus like keratoses.
4.  **bcc** (Basal cell carcinoma): A common variant of epithelial skin cancer that rarely metastasizes.
5.  **akiec** (Actinic keratoses and intraepithelial carcinoma / Bowen's disease): Pre-malignant and non-invasive malignant variations of squamous cell carcinomas.
6.  **vasc** (Vascular lesions): Include angiomas, angiokeratomas, pyogenic granulomas, and hemorrhages.
7.  **df** (Dermatofibroma): A benign skin lesion regarded as a benign proliferation or an inflammatory reaction to minimal trauma.

*(Note: The `nv` class will dominate the dataset. Techniques such as class weighting, focal loss, oversampling (SMOTE), or aggressive data augmentation on minority classes are required to train a robust model.)*

---

## 4. The Pixel Data Files (`hmnist_*.csv`)

These CSV versions of the dataset act as scaled-down, flattened arrays conceptually identical to the famous MNIST digit dataset. They are provided for rapid prototyping and lightweight models.

*   **Structure:** Each row corresponds to an image. The columns represent individual pixel intensities plus the target label at the end.
*   **Example (`hmnist_28_28_RGB.csv`)**: A 28x28 pixel image has 784 pixels. Because it is RGB, it has 3 channels. $784 \\times 3 = 2352$ pixel feature columns. Add the metadata/labels, and you have over 2350 columns per row.
*   **Use-cases**: 
    *   **28x28 RGB/L**: Ideal for simple multi-layer perceptrons (MLP) or quickly testing a basic Convolutional Neural Network (CNN) architecture without writing data-loaders for JPEGs.
    *   **8x8 RGB/L**: Extensively compressed. Mostly useful for non-deep learning models like Random Forests, SVMs, or KNNs.
    *   **L vs RGB**: `L` stands for Luminance (Grayscale). `RGB` retains the color (Red, Green, Blue) data. Color is deeply indicative of skin diseases (e.g., diagnosing melanoma often replies on observing multiple colors), so throwing away color (using 'L') is usually detrimental to high accuracy.

---

## 5. Pre-Processing & Modeling Recommendations

If analyzing or building models from this dataset, adhere to the following checklist:

1.  **Handling Missing Data:** A tiny subset of patients have no recorded `age`. Imputing by median or dropping these rows is acceptable.
2.  **Splitting:** Apply `GroupKFold` using the `lesion_id` column to prevent data bleed between Train/Test splits.
3.  **Class Imbalance:** Due to the heavy skew towards `nv` (Nevi), rely on metrics like Macro-F1 Score, Precision/Recall, or the Balanced Accuracy Score rather than overall raw accuracy. (A model that always guesses `nv` without even looking at the image will still be ~67% accurate).
4.  **Images vs Dataframes:** For high-performance deep learning models (ResNet, EfficientNet, ViT), you should load the raw `.jpg` images from the `HAM10000_images_part_1` and `part_2` directories and resize them to your network's required inputs (e.g., 224x224), ignoring the pre-flattened `hmnist` CSVs entirely.