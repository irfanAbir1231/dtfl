import os
from glob import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from PIL import Image
try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:
    print("PyTorch and Torchvision are required for image transformations.")

def preprocess_metadata(metadata_path, image_dirs):
    """
    Preprocesses the tabular metadata for the HAM10000 dataset.
    """
    print(f"Loading metadata from {metadata_path}...")
    df = pd.read_csv(metadata_path)
    
    # A. Label Encoding: Convert diagnosis string to numeric labels
    print("Applying Label Encoding to 'dx'...")
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["dx"])
    
    # Store the mapping for reference later
    label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Label Mapping: {label_mapping}")
    
    # B. Missing values will be handled after train/test split to prevent data leakage.
    print("Skipping global missing value imputation (will do post-split)...")
    
    # Map image IDs to their absolute file paths (since they are split across two directories)
    print("Mapping image IDs to physical paths...")
    image_paths = {}
    for folder in image_dirs:
        # Search for .jpg files in the provided directories
        for filepath in glob(os.path.join(folder, '*.jpg')):
            filename = os.path.basename(filepath)
            img_id = filename.split('.')[0]
            image_paths[img_id] = filepath
            
    df['path'] = df['image_id'].map(image_paths)
    
    # Check for any unmapped images
    missing_paths = df['path'].isnull().sum()
    if missing_paths > 0:
        print(f"Warning: {missing_paths} images could not be found in the provided directories.")
        
    return df, le

# C. Image Transformations (PyTorch)
def get_transforms():
    """
    Standard image transformations for Deep Learning model pipelines.
    Ready to be used later in the Flower framework federated clients.
    """
    # Transformations for Training (Includes Data Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),       # Standard size for ResNet, MobileNet, ViT, etc.
        transforms.RandomHorizontalFlip(),   # Augmentation
        transforms.RandomVerticalFlip(),     # Augmentation
        transforms.ToTensor(),               # Convert to PyTorch Tensor (scales pixels 0 to 1)
        transforms.Normalize(                # ImageNet standards for pre-trained weights
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Transformations for Validation / Testing (No Augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, test_transform

class SkinLesionDataset(Dataset):
    """
    Custom PyTorch Dataset class to load HAM10000 images efficiently.
    Perfect for integration with Flower framework client data loaders.
    """
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
        
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['path']
        label = row['label']
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    # Base paths
    METADATA_CSV = 'Dataset/HAM10000_metadata.csv'
    IMAGE_DIRS = [
        'Dataset/HAM10000_images_part_1',
        'Dataset/HAM10000_images_part_2'
    ]
    
    # 1. Preprocess Tabular Data
    df_processed, label_encoder = preprocess_metadata(METADATA_CSV, IMAGE_DIRS)
    
    # Save the preprocessed dataframe to a new CSV file
    output_csv = 'Dataset/HAM10000_metadata_preprocessed.csv'
    df_processed.to_csv(output_csv, index=False)
    print(f"\nSaved preprocessed metadata to: {output_csv}")
    
    # Display the first few rows to verify missing value filling, label encoding, and path generation
    print("\nSample of Preprocessed DataFrame:")
    print(df_processed[['image_id', 'dx', 'label', 'age', 'path']].head())
    
    # 2. Prevent Data Leakage: GroupShuffleSplit by lesion_id
    print("\nSplitting dataset into Training and Testing sets (Grouped by lesion_id)...")
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    # Use next() to get the first (and only) split indices
    train_idx, test_idx = next(gss.split(df_processed, groups=df_processed["lesion_id"]))
    
    train_df = df_processed.iloc[train_idx].copy()
    test_df = df_processed.iloc[test_idx].copy()
    
    # Prevent data leakage: Fill missing 'age' using training set median
    if 'age' in train_df.columns:
        train_median_age = train_df['age'].median()
        train_df['age'] = train_df['age'].fillna(train_median_age)
        test_df['age'] = test_df['age'].fillna(train_median_age)
        print(f"Filled missing 'age' with training median: {train_median_age}")
    
    # Save the splits
    train_csv = 'Dataset/HAM10000_metadata_train.csv'
    test_csv = 'Dataset/HAM10000_metadata_test.csv'
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Training set: {len(train_df)} images saved to {train_csv}")
    print(f"Testing set: {len(test_df)} images saved to {test_csv}")
    
    # Verify no leakage happened
    train_lesions = set(train_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])
    leakage = train_lesions.intersection(test_lesions)
    print(f"Lesion overlap between train and test (should be 0): {len(leakage)}")
    
    # 3. Get Transformations
    train_tf, test_tf = get_transforms()
    print("\nTransforms successfully generated.")
    
    # 4. Handle Class Imbalance using Weighted Loss
    print("\nCalculating class weights for CrossEntropyLoss...")
    # Count the instances of each class in the training set
    class_counts = train_df["label"].value_counts().sort_index()
    # Create weights: Inverse frequency (1 / count)
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
    
    # Optional: Normalize weights so they sum to the number of classes or another standard
    # weights = weights / weights.sum() * len(class_counts)
    
    # Initialize the loss function with the calculated weights
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    print("Class Counts:", class_counts.to_dict())
    print("Weights Tensor:", weights)
    
    # Save weights to disk for federated clients to use
    weights_path = 'Dataset/class_weights.pt'
    torch.save(weights, weights_path)
    print(f"Initialized CrossEntropyLoss and saved class weights to {weights_path}.")
    
    # 5. Create Dataset instances for the splits
    train_dataset = SkinLesionDataset(dataframe=train_df, transform=train_tf)
    test_dataset = SkinLesionDataset(dataframe=test_df, transform=test_tf)
    print(f"\nInstantiated PyTorch Train Dataset with {len(train_dataset)} records.")
    print(f"Instantiated PyTorch Test Dataset with {len(test_dataset)} records.")
