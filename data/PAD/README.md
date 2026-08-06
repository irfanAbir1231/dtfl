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

The preprocessing audit removes byte-identical duplicates. If identical
pixels have conflicting diagnoses, every row in that conflict group is
excluded instead of guessing a label; one representative is retained for a
same-label duplicate group. The audit details and affected image IDs are
stored in `PAD_preprocessing_summary.json`.

The train/test split is grouped by `patient_id`, so no patient or lesion can
appear in both sets. It balances diagnosis, diagnosis-by-image-source, and
class-bearing patient counts to limit acquisition-source bias. Training
patients are also kept on exactly one federated client. The default client
partition remains deterministic and non-IID with Dirichlet alpha 0.5, but it
now constrains shard-size imbalance and prevents rare classes from collapsing
to a single example whenever the class count permits.

At runtime, PAD uses 64x64 inputs (HAM10000 is unchanged), PAD training-set RGB
normalization, train-only lesion-preserving augmentation, and moderate
inverse-square-root class/patient sampling. The latter is enabled by default
and can be disabled for an ablation with `--no_pad_balanced_sampling`.

After preparation, start training from the repository root:

```bash
python3 main.py --dataset pad
```

Raw and generated dataset files are intentionally ignored by Git.
