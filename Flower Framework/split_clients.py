import pandas as pd
import numpy as np
import os

def create_non_iid_clients(train_csv_path, num_clients=8, alpha=0.5):
    """
    Splits the dataset into federated clients using a combination of:
    - Label Skew (Dirichlet distribution per class)
    - Quantity Skew (Dirichlet proportions per client)
    While maintaining lesion_id grouping to prevent leakage.
    """
    # Load the training data generated in preprocess.py
    train_df = pd.read_csv(train_csv_path)
    
    # 1. PREVENT LEAKAGE: Group by lesion_id. 
    # We assign LESIONS to clients, not individual images.
    lesion_labels = train_df.groupby('lesion_id')['label'].first()
    unique_labels = lesion_labels.unique()
    
    # Initialize empty clients
    client_lesions = {i: [] for i in range(num_clients)}
    
    # Keep track of available lesions by class
    available_lesions = {c: lesion_labels[lesion_labels == c].index.tolist() for c in unique_labels}
    for c in unique_labels:
        np.random.shuffle(available_lesions[c])
        
    # -------------------------------------------------------------------------
    # CONSTRAINT 1: "At least 3 labels of data will appear strictly in each client"
    # We guarantee this by pre-assigning 1 lesion from 3 random classes to every client.
    # -------------------------------------------------------------------------
    for i in range(num_clients):
        # Pick 3 random, completely unique classes
        chosen_classes = np.random.choice(unique_labels, size=3, replace=False)
        for c in chosen_classes:
            if available_lesions[c]:
                # Pop out 1 physical lesion from that class and give to client
                lesion = available_lesions[c].pop()
                client_lesions[i].append(lesion)

    # -------------------------------------------------------------------------
    # CONSTRAINT 2/3: Non-IID Label Skew + Quantity Skew
    # -------------------------------------------------------------------------
    # Quantity Skew: Create an uneven baseline distribution of how BIG each client is.
    # Some clients will naturally get a massive chunk of data, some very little.
    quantity_proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
    
    for c in unique_labels:
        remaining = available_lesions[c]
        if not remaining:
            continue
            
        # Label Skew: Draw uneven probabilities for how this SPECIFIC class is distributed.
        class_proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        
        # Combine them: This class's skew * the client's overall quantity skew
        combined_proportions = class_proportions * quantity_proportions
        combined_proportions /= combined_proportions.sum() # Normalize to 1.0
        
        # Split the remaining lesions for this class based on those skewed math proportions
        splits = np.split(remaining, (np.cumsum(combined_proportions)[:-1] * len(remaining)).astype(int))
        
        for i in range(num_clients):
            client_lesions[i].extend(splits[i])
            
    # -------------------------------------------------------------------------
    # Generate and Save Client Datasets
    # -------------------------------------------------------------------------
    client_dfs = []
    
    print("\n--- Client Data Distribution Report ---")
    fmt_str = "{:<10} | {:<12} | {:<60}"
    print(fmt_str.format("Client ID", "Total Images", "Label Counts (Class: Count)"))
    print("-" * 90)
    
    for i in range(num_clients):
        # Map the lesions back to their actual multiple image rows
        c_df = train_df[train_df['lesion_id'].isin(client_lesions[i])].copy()
        client_dfs.append(c_df)
        
        # Save the shard
        client_file = f'Dataset/client_{i+1}_train.csv'
        c_df.to_csv(client_file, index=False)
        
        # Calculate stats for the user
        label_counts = c_df['label'].value_counts().sort_index().to_dict()
        label_str = ", ".join([f"{k}: {v}" for k, v in label_counts.items()])
        num_distinct_labels = len(label_counts)
        
        print(fmt_str.format(f"Client {i+1}", len(c_df), label_str))
        
        # Verification Checks
        assert num_distinct_labels >= 3, f"ERROR: Client {i+1} only has {num_distinct_labels} labels!"
        
    # Final check: Did every class get split into at least one client?
    print("\nSuccessfully checked: Every class exists in the datasets.")    
    print(f"Successfully generated {num_clients} client datasets in 'Dataset/' directory.")

if __name__ == "__main__":
    # Seed for reproducibility
    np.random.seed(42) 
    train_csv = 'Dataset/HAM10000_metadata_train.csv'
    
    print(f"Running Non-IID Skewing for {train_csv} (Clients=8)...")
    create_non_iid_clients(train_csv, num_clients=8, alpha=0.5)