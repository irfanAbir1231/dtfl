# DTFL
This repository contains the code for the paper "Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach"

DTFL is a federated learning algorithm designed to speed up training in heterogeneous environments. In heterogeneous environments, the devices participating in federated learning may have different computing resources and network connectivity. DTFL addresses this by dynamically assigning devices to different tiers based on their capabilities.

To do this, this Python program simulates the CPU profile for each client and calculates the intermediate data size and delay. It then uses this information to assign clients to tiers to minimize the overall training time. Detailed information about training time and each client's performance is logged on Weights & Biases ([WandB](https://wandb.ai/)).

DTFL has been shown to achieve significant speedups over traditional federated learning algorithms in heterogeneous environments.

The dataset can be downloaded by running the following command:

```bash
sh download_dataset.sh
```

## Usage

FedAvg is the default aggregation algorithm, so both of the following commands
run the same method:

```bash
python3 main.py
python3 main.py --algorithm fedavg
```

Select another server aggregation algorithm with `--algorithm`:

```bash
# Federated Averaging with server momentum
python3 main.py --algorithm fedavgm

# Adaptive Federated Optimization with Yogi
python3 main.py --algorithm fedyogi

# Adaptive Federated Optimization with Adagrad
python3 main.py --algorithm fedadagrad
```

The algorithm name is case-insensitive. Unsupported names are rejected before
training starts, and the error lists the supported choices.

### Server aggregation options

The supplied defaults allow each method to run using only `--algorithm`:

| Option | Used by | Default |
| --- | --- | ---: |
| `--server_lr` | FedAvgM, FedYogi, FedAdagrad | Per algorithm |
| `--server_momentum` | FedAvgM | `0.9` |
| `--server_beta1` | FedYogi | `0.9` |
| `--server_beta2` | FedYogi | `0.99` |
| `--server_tau` | FedYogi, FedAdagrad | `0.001` |

When `--server_lr` is omitted, it defaults to `1.0` for FedAvgM, `0.01` for
FedYogi, and `0.1` for FedAdagrad. For example, a custom FedYogi run can use:

```bash
python3 main.py \
  --algorithm fedyogi \
  --server_lr 0.005 \
  --server_beta1 0.9 \
  --server_beta2 0.99 \
  --server_tau 0.001
```

The selected aggregation algorithm is compatible with the existing DTFL
tiering, DEATS energy scheduling, privacy mechanism, FedProx option, training
loop, and evaluation pipeline.
