# DTFL
This repository contains the code for the paper "Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach"

DTFL is a federated learning algorithm designed to speed up training in heterogeneous environments. In heterogeneous environments, the devices participating in federated learning may have different computing resources and network connectivity. DTFL addresses this by dynamically assigning devices to different tiers based on their capabilities.

To do this, this Python program simulates the CPU profile for each client and calculates the intermediate data size and delay. It then uses this information to assign clients to tiers to minimize the overall training time. Detailed information about training time and each client's performance is logged on Weights & Biases ([WandB](https://wandb.ai/)).

DTFL has been shown to achieve significant speedups over traditional federated learning algorithms in heterogeneous environments.

The dataset can be downloaded by running the following command:

"
sh download_dataset.sh
"

## Usage

Install the pinned Python 3.9-compatible dependencies:

```bash
python3 -m pip install -r requirements.txt
```

HAM10000 is the default dataset:

```bash
python3 main.py
```

To train with CIFAR-10 instead, use either the standard option or the spelling
used by older commands:

```bash
python3 main.py --dataset cifar10
python3 main.py --datatset cifer
```

The CIFAR-10 loader downloads the dataset into `./data` automatically when it
is not already present.

### PAD-UFES-20

PAD-UFES-20 is a six-class clinical skin-lesion dataset. Download and prepare
it once before training (the Kaggle archive is approximately 3.61 GB):

```bash
python3 -m pip install -r data/PAD/requirements.txt
python3 data/PAD/prepare_pad.py --download
python3 main.py --dataset pad
```

The preparation step creates a patient-disjoint 80/20 train/test split and
eight patient-disjoint, label-skewed federated client shards. HAM10000 remains
the default when `--dataset` is omitted. See [data/PAD/README.md](data/PAD/README.md)
for the generated layout and reproducibility details.
