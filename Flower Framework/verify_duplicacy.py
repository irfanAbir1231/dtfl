import pandas as pd
import hashlib

df = pd.read_csv('Dataset/HAM10000_metadata_preprocessed.csv')
multi_image_lesions = df[df.duplicated(subset=['lesion_id'], keep=False)]

def get_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()

print("Analyzing pixels for all multiple-image lesion cases...")
hashes = [get_hash(p) for p in multi_image_lesions['path']]
unique_hashes = len(set(hashes))

print(f'\n--- IMAGE DUPLICACY REPORT ---')
print(f'Total images belonging to multi-image lesions: {len(hashes)}')
print(f'Unique pixel-level image files (MD5 checks): {unique_hashes}')
print(f'Exact identical copy-pasted files (duplicates): {len(hashes) - unique_hashes}')
