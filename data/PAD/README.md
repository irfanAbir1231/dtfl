# PAD-UFES-20

This directory contains the reproducible download and preprocessing workflow
for the PAD-UFES-20 skin-lesion dataset from Kaggle:

- Dataset: `mahdavi1202/skin-cancer`
- License: CC BY 4.0
- Original publication: Pacheco et al., *Data in Brief* 32 (2020), 106221

Install the downloader and prepare the dataset once:

```bash
python3 -m pip install -r data/PAD/requirements.txt
python3 data/PAD/prepare_pad.py --download
```

The Kaggle archive is approximately 3.61 GB. Raw images are downloaded to
`data/PAD/raw/`. The script then creates:

```text
data/PAD/processed/
├── PAD_metadata_preprocessed.csv
├── PAD_metadata_train.csv
├── PAD_metadata_test.csv
├── client_1_train.csv ... client_8_train.csv
└── PAD_preprocessing_summary.json
```

The train/test split is grouped by `patient_id`, so no patient can appear in
both sets. Training patients are also kept on exactly one federated client.
The default client partition is deterministic, label-skewed, and uses
Dirichlet alpha 0.5. Runtime resizing, augmentation, tensor conversion, and
normalization are performed by the PAD PyTorch loader.

After preparation, start training from the repository root:

```bash
python3 main.py --dataset pad
```

Raw and generated dataset files are intentionally ignored by Git.
