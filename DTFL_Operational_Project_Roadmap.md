# Operational Project Roadmap from Base Paper: Dynamic Tiering-based Federated Learning (DTFL)

**Base paper:** *Speed Up Federated Learning in Heterogeneous Environment: A Dynamic Tiering Approach*  
**Authors:** Seyed Mahmoud Sajjadi Mohammadabadi, Syed Zawad, Feng Yan, Lei Yang  
**arXiv:** 2312.05642v1, cs.LG, 9 Dec 2023  
**Primary system:** **Dynamic Tiering-based Federated Learning (DTFL)**  
**Primary goal:** Speed up federated learning for large models in heterogeneous resource-constrained environments while maintaining accuracy.

---

## 1. Executive Technical Overview

Federated Learning (**FL**) enables multiple clients to collaboratively train a shared global model while keeping raw training data decentralized and private. However, when clients are heterogeneous in **computation capacity**, **communication speed**, and **dataset size**, FL suffers from the **straggler problem**: slower clients dominate each training round, increasing total training time and wasting resources of faster clients.

The paper proposes **Dynamic Tiering-based Federated Learning (DTFL)**, which combines ideas from:

- **Federated Learning**: global aggregation of client-updated models.
- **Split Learning**: partitioning a model into client-side and server-side parts.
- **Tier-based FL**: assigning clients to tiers based on training speed.
- **Local-loss-based split training**: allowing clients and server to train in parallel without waiting for server-side backpropagated gradients.

DTFL assigns clients dynamically to different tiers. Each tier corresponds to a different split point in the global model. Slower clients offload more layers to the server, while faster clients keep more layers locally. The tier assignment is updated by a **dynamic tier scheduler** based on measured historical training time, communication speed, and number of local data batches.

---

## 2. Core Problem Statement

### 2.1 Goal

Collaboratively train a large neural network model, such as **ResNet** or **AlexNet**, using **K clients** operating on heterogeneous resource-constrained devices without centralizing client datasets on the server.

### 2.2 Client Dataset Definition

For client `k`, its local dataset is:

```math
\{(x_i, y_i)\}_{i=1}^{N_k}
```

Where:

- `x_i`: the `i`-th training sample.
- `y_i`: the label associated with `x_i`.
- `N_k`: number of samples in client `k`'s dataset.

### 2.3 Federated Optimization Objective

The standard FL objective is:

```math
\min_w f(w) \overset{def}{=} \min_w \sum_{k=1}^{K} \frac{N_k}{N} \cdot f_k(w)
```

Where:

```math
f_k(w) = \frac{1}{N_k} \sum_{i=1}^{N_k} \ell((x_i, y_i); w)
```

And:

```math
N = \sum_{k=1}^{K} N_k
```

Variables:

- `w`: model parameters.
- `f(w)`: global objective function.
- `f_k(w)`: local objective function of client `k`.
- `ℓ`: loss function.

### 2.4 Main Limitation of Existing FL Methods

Existing FL optimization techniques such as **FedAvg**, **FedProx**, **FedYogi**, and other conventional approaches struggle to efficiently train large models on heterogeneous resource-constrained devices because:

- Every client usually trains the **entire global model**.
- Devices have different CPU/GPU resources.
- Devices have different network bandwidths.
- Clients can have different local dataset sizes.
- Training time varies greatly, producing **stragglers**.
- One slow client can delay the whole round.

### 2.5 DTFL Solution Summary

DTFL introduces:

1. **Multiple tiers**, where each tier corresponds to a different client/server split of the global model.
2. **Dynamic tier scheduling**, assigning clients to appropriate tiers each round.
3. **Local-loss-based split training**, where clients and server update their parts in parallel.
4. **Tier profiling**, estimating per-client training time under different tiers using measured training time, communication speed, transferred data size, and dataset size.
5. **Model aggregation**, combining client-side and server-side updates into full global models.

---

## 3. Background and Related Work

### 3.1 Federated Learning

FL allows decentralized model training without sharing raw data. However, standard FL requires clients to repeatedly download and update the full global model. This becomes inefficient for large models on heterogeneous and resource-constrained devices.

Relevant methods and limitations:

- **FedAvg**: clients train the full global model and aggregate by averaging; suffers from stragglers.
- **Client selection methods**: select fewer clients per round, but may require more rounds.
- **Slowest-client dropping**: neglecting the slowest 30% of clients can mitigate stragglers but requires selecting a sensitive threshold.
- **FedProx**: uses different local epoch numbers but needs careful parameter selection.
- **Tier-based FL methods** such as TiFL and FedAT: group clients by training speed, but still require clients to train the full model.

### 3.2 Split Learning

Split Learning (**SL**) partitions a model into:

- **Client-side model**: initial layers trained locally.
- **Server-side model**: remaining layers trained on server.

SL reduces client-side computation, but classic SL has major drawbacks:

- Clients must wait for server-side gradients during backpropagation.
- Communication of forward activations and backward gradients can be heavy.
- Training can become sequential or synchronization-heavy.

### 3.3 Local-loss-based Split Learning

Local-loss-based split learning uses an **auxiliary network** at the client side to compute a local loss. This allows the client to update its local model without waiting for server backpropagation.

DTFL adopts this idea so that:

- The client trains its client-side model and auxiliary network locally.
- The server trains the server-side model in parallel.
- Communication overhead is reduced compared to classic SL.

### 3.4 Why DTFL Is Different

Existing split-learning or knowledge-transfer approaches often use fixed client-side models. DTFL dynamically adjusts the client-side model size by assigning clients to tiers. This makes DTFL more suitable when client resources change over time.

---

## 4. Notation Reference

| Symbol | Description |
|---|---|
| `K`, `k` | Number and index of clients |
| `x_i`, `y_i` | `i`-th training sample and its label |
| `N_k`, `N` | Size of client `k` dataset and total dataset size |
| `R`, `r` | Number and index of global rounds |
| `A_c,m^(r)` | Number of clients in tier `m` at round `r` |
| `A_c,m^(r)` / `𝒜_c,m^(r)` | Set of clients in tier `m` at round `r` |
| `x_n`, `y_n`, `n` | `n`-th training sample, `n`-th label, datapoint index |
| `w` | Model parameters |
| `d`, `p` | Distance to converged output of client-side model; probability distribution |
| `D` | Distance of the model to the optimal model |
| `Dsize(·)` | Size of transferred data in MB using profiling |
| `m_k^(r)` | Tier of client `k` at round `r` |
| `M` | Number of tiers |
| `𝓜` | Set of tiers |
| `ν` | Communication speed |
| `w_c^m` | Client-side model for tier `m` |
| `w_s^m` | Server-side model for tier `m` |
| `w_a^m` | Auxiliary network for tier `m` |
| `z_i` | Intermediate representation generated by client-side model |
| `η` | Learning rate |
| `T_k^c` | Client-side training time |
| `T_k^com` | Communication time |
| `T_k^s` | Server-side training time |
| `T_k` | Overall per-client training time in a round |
| `T_max` | Straggler training time used by tier scheduler |
| `ν_k^(r)` | Communication speed of client `k` at round `r` |
| `Ñ_k` | Number of data batches of client `k` |

---

## 5. DTFL Methodology Deep-Dive

## 5.1 High-Level DTFL Workflow

At each FL training round:

- [ ] The server profiles clients and estimates training time under possible tiers.
- [ ] The **dynamic tier scheduler** assigns each client to a tier.
- [ ] Each client downloads the client-side model for its assigned tier.
- [ ] Each client performs forward propagation and sends intermediate features plus labels to the server.
- [ ] Each client updates its client-side model using local loss through an auxiliary network.
- [ ] The server updates the corresponding server-side model using the intermediate features.
- [ ] The server reconstructs a complete model for each client from client-side and server-side parts.
- [ ] The server averages complete client models to update the global model.
- [ ] The server updates all tier-specific client-side and server-side models from the new global model.
- [ ] Repeat until convergence or target accuracy.

---

## 5.2 Tiering Local-Loss-Based Training

### 5.2.1 Tier Concept

DTFL divides clients into `M` tiers based on training speed and resource capability.

For tier `m`, the global model `w` is split into:

```math
w = \{w_c^m, w_s^m\}
```

Where:

- `w_c^m`: client-side model for tier `m`.
- `w_s^m`: server-side model for tier `m`.

A lower tier generally means the client holds fewer layers and offloads more computation to the server. A higher tier means the client trains more layers locally.

### 5.2.2 Auxiliary Network

Each client-side model has an auxiliary network:

```math
w_a^m
```

The auxiliary network is connected after the client-side model and is used to compute client-side local loss. In the experiments, it consists of:

- `avgpool` layer.
- Fully connected layer (`f.c.`).

The input dimension of the f.c. layer is adjusted to match the output of the corresponding client-side model.

### 5.2.3 Client-side Objective

For tier `m`, DTFL defines the client-side loss function:

```math
\min_{w_c^m, w_a^m} \sum_{k \in A_c^m} \frac{N_k}{N_m} \cdot f_k^c(w_c^m, w_a^m)
```

Where:

```math
f_k^c(w_c^m, w_a^m) = \frac{1}{N_k} \sum_{i=1}^{N_k} \ell((x_i, y_i); w_c^m, w_a^m)
```

And:

```math
N_m = \sum_{k \in A_c^m} N_k
```

Variables:

- `A_c^m`: set of clients assigned to tier `m`.
- `N_m`: total samples among clients in tier `m`.
- `w_c^{m*}`, `w_a^{m*}`: optimal client-side and auxiliary model parameters.

### 5.2.4 Server-side Objective

Given optimal client-side model `w_c^{m*}`, the server finds `w_s^{m*}` by minimizing:

```math
\min_{w_s^m} \sum_{k \in A_c^m} \frac{N_k}{N_m} \cdot f_k^s(w_s^m, w_c^{m*})
```

Where:

```math
f_k^s(w_s^m, w_c^{m*}) = \frac{1}{N_k} \sum_{i=1}^{N_k} \ell((z_i, y_i); w_s^m)
```

And:

```math
z_i = h_{w_c^{m*}}(x_i)
```

`z_i` is the intermediate output of the client-side model given input `x_i`.

---

## 5.3 Dynamic Tier Scheduling

### 5.3.1 Per-Client Training Time Components

For client `k` in round `r`, assigned to tier `m_k^(r)`:

- `T_k^c(m_k^(r))`: client-side model training time.
- `T_k^com(m_k^(r))`: communication time.
- `T_k^s(m_k^(r))`: server-side model training time.

Since client-side and server-side models train in parallel, the per-client total training time is:

```math
T_k(m_k^{(r)}) = \max\{T_k^c(m_k^{(r)}) + T_k^{com}(m_k^{(r)}),\; T_k^s(m_k^{(r)}) + T_k^{com}(m_k^{(r)})\}
```

### 5.3.2 Scheduling Objective

Because all clients train in parallel, the round time is dominated by the slowest client. DTFL minimizes the maximum client training time:

```math
\min_{\{m_k^{(r)}\}} \max_k T_k(m_k^{(r)}), \quad \text{subject to } \{m_k^{(r)}\} \in \mathcal{M}, \forall k
```

This is an integer programming problem because tier assignments are discrete.

### 5.3.3 Tier Profiling

Before training starts and during training, DTFL estimates training time for clients under different tiers.

#### Profiling Inputs

- Standard data batch.
- Transferred data size for each tier:

```math
Dsize(m_k^{(r)})
```

- Client communication speed:

```math
\nu_k^{(r)}
```

- Number of data batches:

```math
\tilde{N}_k
```

- Historical client-side training time set:

```math
T_k^{cm}
```

#### Communication Time Estimate

For client `k` in tier `m`:

```math
\hat{T}_k^{com}(m_k^{(r+1)}) \leftarrow \frac{Dsize(m_k^{(r)})\tilde{N}_k}{\nu_k^{(r)}}
```

#### Historical Training Time Update

DTFL stores observed client-side training time after subtracting communication time:

```math
T_k^{cm}(m_k^{(r)}) \leftarrow T_k^{cm}(m_k^{(r)}) \cup \left(T_k^{cm}(m_k^{(r)}) - \frac{D_m\tilde{N}_k}{\nu_k^{(r)}}\right)
```

Then it applies **Exponential Moving Average (EMA)**:

```math
\bar{T}_k^{cm}(m_k^{(r)}) \leftarrow EMA(T_k^{cm}(m_k^{(r)}))
```

#### Estimating Training Time for Other Tiers

Since each client only observes training time for its assigned tier, DTFL estimates other tiers using normalized profiling ratios:

```math
\hat{T}_k^c(m_k^{(r+1)}) \leftarrow \frac{T^{cp}(m_k^{(r+1)})}{T^{cp}(m_k^{(r)})} \bar{T}_k^{cm}(m_k^{(r)})
```

Server-side estimate:

```math
\hat{T}_k^s(m_k^{(r+1)}) \leftarrow T^{sp}(m_k^{(r+1)})\tilde{N}_k
```

Where:

- `T^{cp}(m)`: profiled normalized client-side training time for tier `m`.
- `T^{sp}(m)`: profiled normalized server-side training time for tier `m`.

### 5.3.4 Scheduling Rule

First compute the straggler threshold:

```math
T_{max} \leftarrow \max_k \min_m \{\hat{T}_k(m_k^{(r+1)})\}
```

Then assign each client to the highest feasible tier that does not exceed `T_max`:

```math
m_k^{(r+1)} \leftarrow \arg\max_m \left(\{\hat{T}_k(m_k^{(r+1)}) \leq T_{max}\}\right)
```

Interpretation:

- The scheduler first finds the best possible tier for each client.
- It determines the maximum of these best-case times as `T_max`.
- For each client, it chooses the tier with **least offloading** while keeping estimated time under `T_max`.
- This improves resource utilization while controlling stragglers.

---

## 5.4 DTFL Algorithm Pseudocode

```text
Algorithm: DTFL Training Process

MainServer()
1. for each round r = 0 to R - 1 do
2.     m^(r) ← TierScheduler(T^cm(m_k^(r)), ν^(r), Ñ)
3.     for each client k in parallel do
4.         (z_k^(r), y_k) ← ClientUpdate(w_c,m_k^(r))
5.         Measure T_k^cm(m_k^(r)), ν_k^(r), and Ñ_k
6.         Forward propagation of z_k^(r) on w_s,m_k^(r)
7.         Calculate loss and back propagation on w_s,m_k^(r)
8.         w_s,m_k^(r+1) ← w_s,m_k^(r) - η∇f_k^s(w_s^m, w_c^{m*})
9.         Receive updated w_c,m_k^(r+1) from client k
10.        w_k^(r+1) = {w_c,m_k^(r+1), w_s,m_k^(r+1)}
11.    end for
12.    w^(r+1) = (1/K) Σ_k w_k^(r+1)
13.    Update all models (w_c,m^(r+1) and w_s,m^(r+1)) in each tier using w^(r+1)
14. end for

ClientUpdate(w_c,m_k^(r))
15. Forward propagate on local data to calculate z_k^(r)
16. Send (z_k^(r), y_k) to the server
17. Forward propagation to the auxiliary layer
18. Calculate local loss and back propagation
19. w_c,m_k^(r+1) ← w_c,m_k^(r) - η∇f_k^cm(w_c^m(r), w_a^m(r))
20. Send w_c,m_k^(r+1) to the server

TierScheduler(T^cm(m_k^(r)), ν^(r), Ñ)
21. for all client k do
22.     Add (T_k^cm(m_k^(r)) - D_m Ñ_k / ν_k^(r)) into T_k^cm(m_k^(r))
23.     T̄_k^cm(m_k^(r)) ← EMA(T_k^cm(m_k^(r)))
24.     for all tier m_k^(r+1) do
25.         T̂_k^com(m_k^(r+1)) ← Dsize(m_k^(r)) Ñ_k / ν_k^(r)
26.         T̂_k^c(m_k^(r+1)) ← [T^cp(m_k^(r+1)) / T^cp(m_k^(r))] T̄_k^cm(m_k^(r))
27.         T̂_k^s(m_k^(r+1)) ← T^sp(m_k^(r+1)) Ñ_k
28.         Compute T̂_k(m_k^(r+1)) using Equation (5)
29.     end for
30. end for
31. T_max ← max_k min_m {T̂_k(m_k^(r+1))}
32. for all clients k do
33.     m_k^(r+1) ← arg max_m({T̂_k(m_k^(r+1)) ≤ T_max})
34. end for
35. Return m^(r+1)
```

---

## 5.5 Round-Level Implementation Checklist

### Phase A — Initialization

- [ ] Define the global model architecture, e.g., ResNet-56 or ResNet-110.
- [ ] Divide the global model into modules: `md1` to `md8`.
- [ ] Define `M` tiers.
- [ ] For each tier, define:
  - [ ] Client-side modules.
  - [ ] Server-side modules.
  - [ ] Auxiliary `avgpool` and `f.c.` layers.
- [ ] Initialize global model `w`.
- [ ] Initialize tier-specific models `w_c^m`, `w_s^m`, and `w_a^m`.
- [ ] Initialize historical training-time buffers `T_k^cm`.
- [ ] Profile normalized training time per tier:
  - [ ] `T^cp(m)` for client-side.
  - [ ] `T^sp(m)` for server-side.
- [ ] Profile transferred data size per tier `Dsize(m)`.

### Phase B — Per-Round Scheduling

- [ ] Measure or estimate client communication speed `ν_k^(r)`.
- [ ] Measure or store number of local batches `Ñ_k`.
- [ ] Update historical training time for previously assigned tier.
- [ ] Apply EMA to smooth training time estimates.
- [ ] Estimate communication time for all candidate tiers.
- [ ] Estimate client-side training time for all candidate tiers.
- [ ] Estimate server-side training time for all candidate tiers.
- [ ] Compute total estimated time using max of client-side and server-side parallel paths.
- [ ] Compute `T_max = max_k min_m T̂_k(m)`.
- [ ] Assign each client to the largest feasible tier under `T_max`.

### Phase C — Client-Side Training

- [ ] Client downloads assigned `w_c^m`.
- [ ] Client forwards local mini-batches through `w_c^m`.
- [ ] Client produces intermediate features `z_k^(r)`.
- [ ] Client sends `(z_k^(r), y_k)` to server.
- [ ] Client forwards through auxiliary layer `w_a^m`.
- [ ] Client computes local loss.
- [ ] Client performs local backpropagation.
- [ ] Client updates client-side model:

```math
w_{c,m_k}^{(r+1)} \leftarrow w_{c,m_k}^{(r)} - \eta \nabla f_k^{cm}(w_c^{m(r)}, w_a^{m(r)})
```

- [ ] Client sends updated `w_c` to server.

### Phase D — Server-Side Training

- [ ] Server receives intermediate features and labels.
- [ ] Server forwards `z_k^(r)` through assigned `w_s^m`.
- [ ] Server computes server-side loss.
- [ ] Server backpropagates server-side model.
- [ ] Server updates:

```math
w_{s,m_k}^{(r+1)} \leftarrow w_{s,m_k}^{(r)} - \eta \nabla f_k^s(w_s^m, w_c^{m*})
```

### Phase E — Aggregation and Tier Model Refresh

- [ ] Server combines each client's updated client-side and server-side models:

```math
w_k^{(r+1)} = \{w_{c,m_k}^{(r+1)}, w_{s,m_k}^{(r+1)}\}
```

- [ ] Server averages complete client models:

```math
w^{(r+1)} = \frac{1}{K}\sum_k w_k^{(r+1)}
```

- [ ] Server refreshes all tier-specific splits using the new global model:
  - [ ] `w_c,m^(r+1)`
  - [ ] `w_s,m^(r+1)`

---

## 6. Dataset Descriptions

The paper evaluates image classification on four public datasets.

| Dataset | Type | Notes |
|---|---|---|
| CIFAR-10 | Image classification | Used in IID and non-IID settings |
| CIFAR-100 | Image classification | Used in IID and non-IID settings |
| CINIC-10 | Image classification | Used in IID and non-IID settings |
| HAM10000 | Dermatoscopic skin lesion image classification | Used as a medical image dataset |

### 6.1 Non-IID Data Generation

The paper uses **label distribution skew**, where label distributions vary across clients. Non-IID variants are generated using a **Dirichlet distribution** with:

```text
Concentration parameter = 0.5
Random seed = fixed
```

The distribution is fixed to ensure fair comparisons among methods.

### 6.2 Why LEAF Benchmark Was Not Used

The paper does not use LEAF benchmark datasets because:

- The datasets are either too small, or
- Too simple for large CNN models, and therefore unsuitable for evaluating DTFL with large CNNs.

### 6.3 CIFAR-10 Non-IID Distribution for 10 Clients

| Client | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8 | c9 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| k=0 | 372 | 2398 | 518 | 2 | 1036 | 641 | 210 | 0 | 0 | 0 | 5177 |
| k=1 | 191 | 84 | 77 | 1008 | 917 | 305 | 0 | 263 | 1295 | 736 | 4876 |
| k=2 | 23 | 362 | 1281 | 40 | 358 | 1011 | 123 | 451 | 316 | 284 | 4249 |
| k=3 | 97 | 1032 | 289 | 185 | 670 | 0 | 178 | 84 | 1048 | 1467 | 5050 |
| k=4 | 40 | 130 | 1209 | 186 | 5 | 57 | 3307 | 1176 | 0 | 0 | 6110 |
| k=5 | 1639 | 0 | 296 | 121 | 68 | 717 | 403 | 372 | 1932 | 0 | 5548 |
| k=6 | 1 | 866 | 60 | 101 | 451 | 598 | 120 | 83 | 323 | 2316 | 4919 |
| k=7 | 1307 | 15 | 428 | 0 | 290 | 2 | 356 | 1448 | 50 | 192 | 4088 |
| k=8 | 849 | 0 | 88 | 910 | 1187 | 1414 | 24 | 229 | 36 | 4 | 4741 |
| k=9 | 481 | 113 | 754 | 2447 | 18 | 225 | 279 | 894 | 0 | 1 | 5242 |

---

## 7. Experimental Setup

### 7.1 Baselines

DTFL is compared against:

| Method | Description / Role |
|---|---|
| FedAvg | Standard federated averaging baseline |
| SplitFed | Split learning combined with FL |
| FedYogi | Adaptive federated optimization baseline |
| FedGKT | Group Knowledge Transfer; clients train small models and transfer knowledge |

The paper does not compare with:

- **FedProx**, because it performs worse than FedAvg in large CNN settings according to the cited FedGKT rationale.
- **FedMA**, because it cannot work on modern DNNs containing batch normalization layers, such as ResNet.

### 7.2 Software and Hardware

| Component | Setup |
|---|---|
| Language | Python 3.11.3 |
| Library | PyTorch 1.13.1 |
| Code availability | GitHub implementation by authors |
| Server CPU | Dual-socket Intel Xeon CPU E5-2630 v4 @ 2.20 GHz |
| Hyper-threading | Disabled |
| GPUs | 4 × NVIDIA GeForce GTX 1080 Ti |
| Memory | 64 GB |

### 7.3 Simulated Heterogeneous Resource Profiles

Each client is assigned a simulated CPU and communication profile. Profiles can change during training to simulate dynamic environments.

| Profile | CPU Capacity | Network Speed |
|---|---:|---:|
| P1 | 4 CPUs | 100 Mbps |
| P2 | 2 CPUs | 30 Mbps |
| P3 | 1 CPU | 30 Mbps |
| P4 | 0.2 CPU | 30 Mbps |
| P5 | 0.1 CPU | 10 Mbps |

### 7.4 Dynamic Environment Setup

In Table 3 experiments:

- 20% of clients are initially assigned to each resource profile.
- Every **50 rounds**, profiles of **30% of clients** are randomly changed.
- All clients participate in every round.

For number-of-tiers experiments:

- Clients' CPU profiles switch randomly every **20 rounds** within the same case profile set.

### 7.5 Optimizer and Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimizer | ADAM |
| Initial learning rate for CIFAR-10 | 0.001 |
| Initial learning rate for CIFAR-100 | 0.001 |
| Initial learning rate for CINIC-10 | 0.001 |
| Initial learning rate for HAM10000 | 0.0001 |
| LR schedule | Reduce by factor 0.9 after accuracy reaches plateau |
| Local batch size for 200 clients | 50 |
| Local batch size for other experiments | 100 |
| Local epoch | 1 |

---

## 8. Model Architecture

The paper evaluates **ResNet-56** and **ResNet-110**. The models are divided into modules `md1` to `md8`; tiers are defined by choosing different split positions among these modules.

### 8.1 Module-Level Design Principle

- `md1` to `md7` contain convolutional residual blocks.
- `md8` contains final average pooling and fully connected classification layer.
- For each client-side split, DTFL adds an auxiliary `avgpool` and `f.c.` layer.
- For SplitFed baseline, the global model is split after `md2`.
- For FedGKT, the paper follows the settings of He et al. (2020a).

### 8.2 ResNet-56 Architecture

| Module | Parameters and Shape | Repetition |
|---|---|---:|
| md1 | `conv1: 3×16×3×3, stride=(1,1), padding=(1,1)`; `maxpool: 3×1` | ×1 |
| md2 | `conv1: 16×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)`; `downsample.conv: 16×64×1×1, stride=(1,1)` | ×1 |
| md2 | `conv1: 64×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)` | ×2 |
| md3 | `conv1: 64×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)` | ×3 |
| md4 | `conv1: 64×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)`; `downsample.conv: 64×128×1×1, stride=(2,2)` | ×1 |
| md4 | `conv1: 128×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)` | ×2 |
| md5 | `conv1: 128×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)` | ×3 |
| md6 | `conv1: 128×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)`; `downsample.conv: 128×256×1×1, stride=(2,2)` | ×1 |
| md6 | `conv1: 256×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)` | ×2 |
| md7 | `conv1: 256×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)` | ×3 |
| md8 | `avgpool`; `fc: 256×10` | ×1 each |

### 8.3 ResNet-110 Architecture

| Module | Parameters and Shape | Repetition |
|---|---|---:|
| md1 | `conv1: 3×16×3×3, stride=(1,1), padding=(1,1)`; `maxpool: 3×1` | ×1 |
| md2 | `conv1: 16×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)`; `downsample.conv: 16×64×1×1, stride=(1,1)` | ×1 |
| md2 | `conv1: 64×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)` | ×5 |
| md3 | `conv1: 64×16×1×1, stride=(1,1)`; `conv2: 16×16×3×3, stride=(1,1), padding=(1,1)`; `conv3: 16×64×1×1, stride=(1,1)` | ×6 |
| md4 | `conv1: 64×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)`; `downsample.conv: 64×128×1×1, stride=(2,2)` | ×1 |
| md4 | `conv1: 128×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)` | ×5 |
| md5 | `conv1: 128×32×1×1, stride=(1,1)`; `conv2: 32×32×3×3, stride=(1,1), padding=(1,1)`; `conv3: 32×128×1×1, stride=(1,1)` | ×6 |
| md6 | `conv1: 128×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)`; `downsample.conv: 128×256×1×1, stride=(2,2)` | ×1 |
| md6 | `conv1: 256×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)` | ×5 |
| md7 | `conv1: 256×64×1×1, stride=(1,1)`; `conv2: 64×64×3×3, stride=(1,1), padding=(1,1)`; `conv3: 64×256×1×1, stride=(1,1)` | ×6 |
| md8 | `avgpool`; `fc: 256×10` | ×1 each |

---

## 9. Tier Architectures

### 9.1 Seven-Tier Architecture (`M = 7`)

Client-side models include `avgpool` and `f.c.` as auxiliary layers.

| Tier | Client-side modules | Auxiliary f.c. | Server-side modules |
|---:|---|---|---|
| 1 | md1 | 16×10 | md2, md3, md4, md5, md6, md7, md8 |
| 2 | md1, md2 | 64×10 | md3, md4, md5, md6, md7, md8 |
| 3 | md1, md2, md3 | 64×10 | md4, md5, md6, md7, md8 |
| 4 | md1, md2, md3, md4 | 128×10 | md5, md6, md7, md8 |
| 5 | md1, md2, md3, md4, md5 | 128×10 | md6, md7, md8 |
| 6 | md1, md2, md3, md4, md5, md6 | 256×10 | md7, md8 |
| 7 | md1, md2, md3, md4, md5, md6, md7 | 256×10 | md8 |

### 9.2 Varying Number of Tiers

| # Tiers | Tier | Client-side | Server-side |
|---:|---:|---|---|
| 1 | 1 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 2 | 1 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 2 | 2 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 3 | 1 | md1, md2, md3, md4, md5 | md6, md7, md8 |
| 3 | 2 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 3 | 3 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 4 | 1 | md1, md2, md3, md4 | md5, md6, md7, md8 |
| 4 | 2 | md1, md2, md3, md4, md5 | md6, md7, md8 |
| 4 | 3 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 4 | 4 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 5 | 1 | md1, md2, md3 | md4, md5, md6, md7, md8 |
| 5 | 2 | md1, md2, md3, md4 | md5, md6, md7, md8 |
| 5 | 3 | md1, md2, md3, md4, md5 | md6, md7, md8 |
| 5 | 4 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 5 | 5 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 6 | 1 | md1, md2 | md3, md4, md5, md6, md7, md8 |
| 6 | 2 | md1, md2, md3 | md4, md5, md6, md7, md8 |
| 6 | 3 | md1, md2, md3, md4 | md5, md6, md7, md8 |
| 6 | 4 | md1, md2, md3, md4, md5 | md6, md7, md8 |
| 6 | 5 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 6 | 6 | md1, md2, md3, md4, md5, md6, md7 | md8 |
| 7 | 1 | md1 | md2, md3, md4, md5, md6, md7, md8 |
| 7 | 2 | md1, md2 | md3, md4, md5, md6, md7, md8 |
| 7 | 3 | md1, md2, md3 | md4, md5, md6, md7, md8 |
| 7 | 4 | md1, md2, md3, md4 | md5, md6, md7, md8 |
| 7 | 5 | md1, md2, md3, md4, md5 | md6, md7, md8 |
| 7 | 6 | md1, md2, md3, md4, md5, md6 | md7, md8 |
| 7 | 7 | md1, md2, md3, md4, md5, md6, md7 | md8 |

---

## 10. Evaluation Metrics

The paper primarily evaluates:

| Metric | Meaning |
|---|---|
| Training time in seconds | Time needed to reach target accuracy |
| Server test accuracy | Accuracy of global/server model during training |
| Target accuracy | Dataset-specific accuracy threshold used to compare methods |
| Computation time | Time spent training client/server models |
| Communication time | Time spent transferring model parameters/intermediate data |
| Overall training time | Computation time + communication time or round-level completion time |
| Model accuracy under privacy methods | Accuracy after integrating distance correlation or patch shuffling |

### 10.1 Target Accuracy Thresholds Used in Table 3

| Dataset | Setting | Target Accuracy |
|---|---|---:|
| CIFAR-10 | IID | 80% |
| CIFAR-10 | non-IID | 70% |
| CIFAR-100 | IID | 55% |
| CIFAR-100 | non-IID | 50% |
| CINIC-10 | IID | 75% |
| CINIC-10 | non-IID | 65% |
| HAM10000 | Not separately listed as IID/non-IID in table | 75% |

---

## 11. Key Experimental Results

### 11.1 Tier-Level Training Time: ResNet-110, CIFAR-10 IID, 10 Clients, `M = 6`

Goal: reach 80% accuracy. In each experiment, all clients are assigned to the same tier.

#### Case 1 Profiles

- 2 CPUs with 30 Mbps.
- 1 CPU with 30 Mbps.
- 0.2 CPU with 30 Mbps.

#### Case 2 Profiles

- 4 CPUs with 100 Mbps.
- 1 CPU with 30 Mbps.
- 0.1 CPU with 10 Mbps.

| Case | Metric | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 | FedAvg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Case 1 | Computation Time | 4622 | 8106 | 9982 | 10681 | 11722 | 12250 | 13396 |
| Case 1 | Communication Time | 5911 | 5995 | 2187 | 2189 | 1018 | 908 | 16 |
| Case 1 | Overall Training Time | 10533 | 14101 | 12170 | 12871 | 12741 | 13158 | 13408 |
| Case 2 | Computation Time | 8384 | 14634 | 17993 | 19027 | 21428 | 22344 | 24428 |
| Case 2 | Communication Time | 17754 | 18090 | 6720 | 6762 | 2941 | 2653 | 43 |
| Case 2 | Overall Training Time | 26138 | 32724 | 24713 | 25989 | 24369 | 24997 | 24471 |

Key interpretation:

- Lower tiers reduce client computation but can increase communication.
- Higher tiers reduce communication but increase local computation.
- The optimal tier is non-trivial and depends on client compute, network speed, and dataset size.

### 11.2 Normalized Training Time Across Tiers: ResNet-56, 10 Clients

Normalized relative to Tier 1.

| Metric | Tier 1 | Tier 2 | Tier 3 | Tier 4 | Tier 5 | Tier 6 |
|---|---:|---:|---:|---:|---:|---:|
| Client-side Training Time | 1.00 ± 0.04 | 1.63 ± 0.10 | 2.16 ± 0.15 | 2.68 ± 0.22 | 3.30 ± 0.24 | 3.81 ± 0.28 |
| Server-side Training Time | 1.00 ± 0.07 | 0.82 ± 0.06 | 0.65 ± 0.06 | 0.51 ± 0.04 | 0.33 ± 0.03 | 0.20 ± 0.01 |

Key interpretation:

- As tier number increases, client-side computation increases.
- As tier number increases, server-side computation decreases.
- Normalized ratios are stable because they depend on model sizes under each tier, not the client's current resource profile.

### 11.3 Training Time Comparison with Baselines: 10 Clients, `M = 7`

Training time in seconds to reach the target accuracy.

| Method | Global Model | CIFAR-10 IID | CIFAR-10 non-IID | CIFAR-100 IID | CIFAR-100 non-IID | CINIC-10 IID | CINIC-10 non-IID | HAM10000 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| DTFL | ResNet-56 | 2750 | 3986 | 3585 | 6093 | 23968 | 40138 | 2353 |
| DTFL | ResNet-110 | 4816 | 7054 | 5678 | 9874 | 42099 | 70469 | 3615 |
| FedAvg | ResNet-56 | 13157 | 20773 | 19170 | 35350 | 114509 | 197926 | 11566 |
| FedAvg | ResNet-110 | 24471 | 39094 | 36360 | 66317 | 210468 | 395423 | 22328 |
| SplitFed | ResNet-56 | 35877 | 46514 | 54174 | 97859 | 271873 | 510156 | 19549 |
| SplitFed | ResNet-110 | 67265 | 84342 | 101783 | 183122 | 521334 | 896627 | 43581 |
| FedYogi | ResNet-56 | 9122 | 13130 | 12727 | 19216 | 82083 | 113464 | 8071 |
| FedYogi | ResNet-110 | 19299 | 25668 | 23978 | 35356 | 155212 | 219134 | 14932 |
| FedGKT | ResNet-56 | 25458 | 30808 | 36838 | 59461 | 184589 | 218065 | 37181 |
| FedGKT | ResNet-110 | 39676 | 47458 | 64457 | 98754 | 321534 | 411259 | 61755 |

Key findings:

- DTFL significantly reduces training time in every reported IID and non-IID case.
- For CIFAR-10 IID with ResNet-110, DTFL reduces FedAvg training time by approximately 80% to reach target accuracy.
- DTFL converges faster than FedAvg, FedYogi, FedGKT, and SplitFed in the reported training curves.

### 11.4 Scalability with Different Numbers of Clients

Dataset: CIFAR-10 IID  
Model: ResNet-110  
Target accuracy: 80%  
Client participation: 10% of all clients sampled per round.

| # Clients | DTFL | FedAvg | SplitFed | FedYogi | FedGKT |
|---:|---:|---:|---:|---:|---:|
| 20 | 1877 | 7950 | 21350 | 6341 | 14595 |
| 50 | 2547 | 10435 | 29026 | 8073 | 17872 |
| 100 | 3102 | 14032 | 36449 | 10760 | 24438 |
| 200 | 3594 | 16060 | 43942 | 12786 | 27632 |

Key finding:

- Increasing the number of clients does not adversely affect DTFL relative to baselines; DTFL remains significantly faster.

### 11.5 Impact of Number of Tiers

Experiment:

- Dataset: CIFAR-10 IID.
- Clients: 10.
- Model: ResNet-110.
- Target accuracy: 80%.
- Number of tiers tested: 1 to 7.
- Client CPU profiles randomly switch every 20 training rounds.

Key findings:

- Training time generally decreases as the number of tiers increases.
- More tiers provide more flexibility to tune the client/server split for heterogeneous clients.
- Tier design must be careful: arbitrary splits can reduce model accuracy.
- For ResNet-110, the authors found 7 tiers to significantly reduce training time while maintaining accuracy.

---

## 12. Privacy Discussion and Privacy Extension Tasks

### 12.1 Privacy Risk

DTFL exchanges hidden feature maps/intermediate representations:

```math
z_i
```

This can leak information under strong attackers, especially through model inversion or feature-map reconstruction attacks.

### 12.2 Threats Discussed

#### Threat 1 — Model Inversion Attacks

Attackers attempt to reconstruct client data from feature maps or model parameter transfers.

The paper notes that prior work shows attackers need access to all model parameters or gradients to recover data. DTFL transmits partial/fragmented models, making this harder.

Mitigation:

- Use separate servers for:
  - model aggregation, and
  - model training.

This prevents one server from accessing all model parameters and intermediate data.

#### Threat 2 — Replicating Client Model Through Dummy Data

An attacker may input dummy data into the client local model and train a replica using resulting feature maps.

Mitigation:

- Deny clients access to:
  - external datasets,
  - query services,
  - dummy data.

### 12.3 Compatible Privacy Methods

DTFL can integrate existing privacy-preserving techniques:

- Distance correlation.
- Differential privacy.
- Patch shuffling.
- PixelDP.
- SplitGuard.
- Cryptography techniques.

### 12.4 Distance Correlation Regularization

The paper suggests adding a regularization term to reduce mutual information between raw input and intermediate feature map.

Private client-side objective:

```math
f_k^{c,private}(w_c^m, w_a^m) = (1 - \alpha)f_k^c(w_c^m, w_a^m) + \alpha DCor(x_i, z_i)
```

Where:

- `α`: balances model performance and privacy.
- `DCor`: distance correlation.
- `x_i`: raw input.
- `z_i`: intermediate feature map.

### 12.5 Privacy Accuracy Results

Dataset: CIFAR-10  
Model: ResNet-56  
Clients: 20

| Method | Distance Correlation α=0.00 | α=0.25 | α=0.50 | α=0.75 | Patch Shuffling |
|---|---:|---:|---:|---:|---:|
| Accuracy | 87.1 | 86.8 | 83.5 | 75.6 | 85.4 |

Key findings:

- Increasing `α` improves privacy but reduces accuracy.
- Small `α` values can improve privacy without significant accuracy loss.
- Patch shuffling applied to intermediate data has minimal accuracy impact under the reported setting.
- The server does not know client-specific `α` values, and `α` can vary across clients, helping prevent inference.

### 12.6 Privacy Implementation Checklist

- [ ] Implement optional distance correlation loss term.
- [ ] Add hyperparameter `alpha` per client.
- [ ] Allow client-specific `alpha` values unknown to server.
- [ ] Compute `DCor(x_i, z_i)` between raw input and intermediate feature maps.
- [ ] Update client loss:

```math
loss = (1 - alpha) * client_task_loss + alpha * distance_correlation(x, z)
```

- [ ] Implement optional patch shuffling on intermediate data.
- [ ] Evaluate accuracy/privacy trade-off for `α ∈ {0.00, 0.25, 0.50, 0.75}`.
- [ ] Log accuracy after privacy integration.
- [ ] Log reconstruction-risk proxy metrics if extending the paper.

---

## 13. Convergence Analysis

DTFL convergence is shown for both client-side and server-side models under convex and non-convex losses.

### 13.1 Assumptions

#### A1 — L-smoothness

The loss function is differentiable and L-smooth:

```math
\|\nabla f(u) - \nabla f(v)\| \leq L\|u - v\|, \quad \forall f,u,v
```

#### A2 — Bounded Gradients

Expected squared norm of stochastic gradients is bounded:

```math
\mathbb{E}\|\nabla f(u)\|^2 \leq G_1^2, \quad \forall f,u
```

#### A3 — Bounded Variance

Stochastic gradient is unbiased:

```math
g_i(u) := \nabla f(u), \quad \mathbb{E}[g_i(u)] = \nabla f_i(u)
```

Variance is bounded:

```math
\mathbb{E}[|g_i(u) - \nabla f_i(u)|^2] \leq \sigma^2, \quad \forall f,u
```

#### A4 — μ-convexity

For `μ ≥ 0`:

```math
f(u) + \nabla^T f(u)(v-u) + \frac{\mu}{2}\|v-u\|^2 \leq f(v), \quad \forall f,u,v
```

#### A5 — Bounded Gradient Dissimilarity

For both client and server sides and all tier models, constants `G_2 ≥ 0` and `B ≥ 1` exist such that:

```math
\frac{1}{K}\sum_{i=1}^{K}\|\nabla f_i(u)\|^2 \leq G_2^2 + B^2\|\nabla f(u)\|^2, \quad \forall u
```

For convex `{f_i}`, this relaxes to:

```math
\frac{1}{K}\sum_{i=1}^{K}\|\nabla f_i(u)\|^2 \leq G_2^2 + 2LB^2(f(u)-f^*), \quad \forall u
```

#### A6 — Bounded Distance

The time-varying parameter satisfies:

```math
d_m^{c(r)} < \infty, \quad \forall m,r
```

### 13.2 Theorem 1 — Convergence of DTFL

Under assumptions **A1**, **A2**, **A3**, and **A5**, DTFL convergence properties are:

#### Convex Case

Under **A4**, if:

```math
\eta \leq \frac{1}{8L(1+B^2)}
```

and

```math
R \geq \frac{4L(1+B^2)}{\mu}
```

Client-side model convergence rate:

```math
O\left(\mu D^2 \exp\left(-\frac{\eta}{2}\mu R\right) + \frac{\eta H_1^2}{\mu R A_m}\right)
```

Server-side model convergence rate:

```math
O\left(\frac{C_1}{R} + \frac{H_2\sqrt{F_{0m}^s}}{\sqrt{R A_m}} + \frac{F_{0m}^s}{\eta_{max}R}\right)
```

#### Non-convex Case

If both `f_c^m` and `f_s^m` are non-convex with:

```math
\eta \leq \frac{1}{8L(1+B^2)}
```

Client-side model convergence rate:

```math
O\left(\frac{H_1\sqrt{F_{0m}^c}}{\sqrt{R A_m}} + \frac{F_{0m}^c}{\eta_{max}R}\right)
```

Server-side model convergence rate:

```math
O\left(\frac{C_2}{R} + \frac{H_2\sqrt{F_{0m}^s}}{\sqrt{R A_m}} + \frac{F_{0m}^s}{\eta_{max}R}\right)
```

Where:

```math
H_1^2 := \sigma^2 + \left(1 - \frac{A_m}{K}\right)G_2^2
```

```math
H_2^2 := L^3(B^2+1)F_{0m}^s + \left(1 - \frac{A_m}{K}\right)L^2G_2^2
```

```math
D := \|w_{0m}^c - w_m^{c*}\|
```

```math
F_{0m}^c := f_m^c(w_{0m}^c)
```

```math
F_{0m}^s := f_m^s(w_{0m}^s)
```

```math
C_1 = G_1\sqrt{G_2^2 + 2LB^2F_{0m}^s}\sum_r d_m^{c(r)}
```

```math
C_2 = G_1\sqrt{G_2^2 + B^2G_1^2}\sum_r d_m^{c(r)}
```

```math
A_m = \min_r\{A_{c,m}^{(r)} > 0\}
```

`d_m^{c(r)}` denotes the distance between the density function of the client-side model output and its converged state.

### 13.3 Practical Interpretation

- Client-side and server-side models converge as global rounds `R` increase.
- Convergence differs by tier because `A_m` can differ by tier.
- More clients in a tier can improve convergence for that tier.
- Server-side convergence depends on client-side convergence because server input distributions depend on intermediate representations generated by client-side models.

---

## 14. Proof Appendix: Key Lemmas and Derivation Notes

### 14.1 Lemma 1 — Linear Convergence Rate

For every non-negative sequence `{d_{r-1}}_{r≥1}` and parameters `μ > 0`, `η_max ∈ (0, 1/μ]`, `q ≥ 0`, and `R ≥ 1/(2η_max μ)`, there exists a constant step size `η ≤ η_max` and weights:

```math
w_r := (1 - \mu\eta)^{1-r}
```

such that for:

```math
W_R := \sum_{r=1}^{R+1} w_r
```

```math
\Psi_R := \frac{1}{W_R}\sum_{r=1}^{R+1}\left(\frac{w_r}{\eta}(1-\mu\eta)d_{r-1} - \frac{w_r}{\eta}d_r + q\eta w_r\right)
```

The result is:

```math
\Psi_R = O\left(\mu d_0 \exp(-\mu\eta_{max}R) + \frac{q}{\mu R}\right)
```

### 14.2 Lemma 2 — Non-convex Convergence Rate

For non-negative `{d_{r-1}}`, `η_max ≥ 0`, `q ≥ 0`, `R ≥ 0`, there exists `η ≤ η_max` and weights `w_r = 1` such that:

```math
\Psi_R := \frac{1}{R+1}\sum_{r=1}^{R+1}\left(\frac{d_{r-1}}{\eta} - \frac{d_r}{\eta} + q_1\eta + q_2\eta^2\right)
```

```math
\leq \frac{d_0}{\eta_{max}(R+1)} + \frac{2\sqrt{q_1d_0}}{\sqrt{R+1}} + 2\left(\frac{d_0}{R+1}\right)^{2/3}q_2^{1/3}
```

### 14.3 Lemma 3 — Relaxed Triangle Inequality

For vectors `{v_1, ..., v_τ}`:

```math
\|v_i + v_j\|^2 \leq (1+a)\|v_i\|^2 + \left(1 + \frac{1}{a}\right)\|v_j\|^2, \quad a > 0
```

```math
\left\|\sum_{i=1}^{\tau} v_i\right\|^2 \leq \tau\sum_{i=1}^{\tau}\|v_i\|^2
```

### 14.4 Lemma 4 — Separating Mean and Variance

For random variables `{Ξ_1, ..., Ξ_κ}` with mean `E[Ξ_i]=ξ_i` and variance bounded by `σ²`:

```math
\mathbb{E}\left[\left\|\sum_{i=1}^{\kappa}\Xi_i\right\|^2\right] \leq \left\|\sum_{i=1}^{\kappa}\xi_i\right\|^2 + \kappa^2\sigma^2
```

For conditional mean:

```math
\mathbb{E}[\Xi_i | \Xi_{i-1}, ..., \Xi_1] = \xi_i
```

Then:

```math
\mathbb{E}\left[\left\|\sum_{i=1}^{\kappa}\Xi_i\right\|^2\right] \leq 2\left\|\sum_{i=1}^{\kappa}\xi_i\right\|^2 + 2\kappa\sigma^2
```

### 14.5 Lemma 5 — Perturbed Strong Convexity

For any L-smooth and μ-strongly convex function `h`, and any `u`, `v`, `w`:

```math
\langle \nabla h(u), w-v \rangle \geq h(w)-h(v) + \frac{L}{4}\|v-w\|^2 - L\|w-u\|^2
```

### 14.6 Client-side Convex Derivation: Key Update

The client-side model update is:

```math
\Delta w_c^m = -\frac{\eta}{A_{c,m}^{(r)}}\sum_{k \in A_{c,m}^{(r)}} g_k^{cm}(w_{k}^{cm})
```

Expectation:

```math
\mathbb{E}[\Delta w_c^m] = -\eta \sum_k^K \mathbb{E}[\nabla f_k^{cm}(w_k^{cm})]
```

The proof assumes one local epoch for simplicity, matching the experiments.

Client-side convergence bound:

```math
\mathbb{E}[f^{cm}(w_m^{cR})] - f^{cm}(w_m^{c*})
\leq \|w_{m}^{c0} - w_m^{c*}\|^2\mu\exp\left(-\frac{\eta}{2}\mu R\right)
+ \frac{\eta}{\mu R}\left(\frac{\sigma^2}{A_m} + \left(1-\frac{A_m}{K}\right)\frac{4G_2^2}{A_m}\right)
```

Asymptotic form:

```math
\mathbb{E}[f^{cm}(w_m^{cR})] - f^{cm}(w^{c*}) = O\left(\mu D^2\exp\left(-\frac{\eta}{2}\mu R\right) + \frac{\eta H_1^2}{\mu R A_m}\right)
```

### 14.7 Server-side Convergence: Distribution Shift Term

Server-side convergence depends on the distribution of client-side outputs. Define:

```math
z_m^{c(r)} \sim p_m^{c(r)}(z)
```

and converged density:

```math
p_m^{c(*)}(z)
```

Distance term:

```math
d_m^{c(r)} \triangleq \int |p_m^{c(r)}(z) - p_m^{c(*)}(z)|dz
```

The server-side proof includes the convergence of:

```math
\sum_r d_m^{c(r)}
```

under assumption A6.

---

## 15. Development Team Implementation Plan

## 15.1 Repository Structure Recommendation

```text
dtfl_project/
  configs/
    cifar10_resnet56.yaml
    cifar10_resnet110.yaml
    ham10000_resnet56.yaml
  data/
    partition.py
    noniid_dirichlet.py
    ham10000_loader.py
  models/
    resnet_modules.py
    tier_splitter.py
    auxiliary_heads.py
  privacy/
    distance_correlation.py
    patch_shuffle.py
  scheduler/
    profiler.py
    tier_scheduler.py
    ema.py
  federated/
    client.py
    server.py
    aggregation.py
    dtfl_trainer.py
  experiments/
    run_dtfl.py
    run_fedavg.py
    run_fedyogi.py
    run_splitfed.py
    run_fedgkt.py
  logs/
  tests/
```

## 15.2 Core Modules to Implement

### A. Model Modules

- [ ] Implement ResNet-56 with modules `md1` to `md8`.
- [ ] Implement ResNet-110 with modules `md1` to `md8`.
- [ ] Implement tier-based model splitter.
- [ ] Implement auxiliary heads:
  - [ ] `avgpool`
  - [ ] `f.c.` matching output dimension: `16×10`, `64×10`, `128×10`, `256×10`.

### B. Data Pipeline

- [ ] Load CIFAR-10.
- [ ] Load CIFAR-100.
- [ ] Load CINIC-10.
- [ ] Load HAM10000.
- [ ] Implement IID split.
- [ ] Implement non-IID Dirichlet label skew.
- [ ] Use concentration parameter `0.5`.
- [ ] Use fixed random seed for fair comparison.

### C. Client Logic

- [ ] Receive assigned tier.
- [ ] Download correct client-side model.
- [ ] Run forward pass to compute `z`.
- [ ] Send `(z, y)` to server.
- [ ] Compute auxiliary local loss.
- [ ] Backpropagate locally.
- [ ] Update client-side model.
- [ ] Return updated client-side weights.
- [ ] Report training time and communication speed.

### D. Server Logic

- [ ] Maintain global model.
- [ ] Maintain tier-specific split models.
- [ ] Receive intermediate features and labels.
- [ ] Run server-side forward/backward.
- [ ] Update server-side model.
- [ ] Combine client-side and server-side model for each client.
- [ ] Aggregate full models using FedAvg-style averaging.
- [ ] Refresh tier models from new global model.

### E. Dynamic Tier Scheduler

- [ ] Track historical client-side training times.
- [ ] Track communication speed per client per round.
- [ ] Track local number of batches.
- [ ] Profile transferred data size per tier.
- [ ] Profile normalized client-side/server-side times.
- [ ] Apply EMA smoothing.
- [ ] Estimate all candidate tier times.
- [ ] Compute `T_max`.
- [ ] Assign largest feasible tier under `T_max`.

### F. Experiment Runner

- [ ] Reproduce Table 3 training-time comparisons.
- [ ] Reproduce client scalability experiment from Table 4.
- [ ] Reproduce tier-number sensitivity experiment from Figure 3.
- [ ] Reproduce privacy trade-off from Table 5.

---

## 16. Required Logs and Experiment Outputs

Track the following per round:

- [ ] Round index `r`.
- [ ] Client ID `k`.
- [ ] Assigned tier `m_k^(r)`.
- [ ] Client-side training time.
- [ ] Server-side training time.
- [ ] Communication time.
- [ ] Estimated total time.
- [ ] Actual total time.
- [ ] `T_max`.
- [ ] Communication speed `ν_k^(r)`.
- [ ] Number of batches `Ñ_k`.
- [ ] Local loss.
- [ ] Server-side loss.
- [ ] Global test accuracy.
- [ ] Target accuracy reached flag.
- [ ] Time to target accuracy.
- [ ] Privacy alpha value if used.
- [ ] Distance correlation value if used.

---

## 17. Reproduction Targets

### 17.1 Main Reproduction Table

- [ ] Reproduce DTFL vs FedAvg, SplitFed, FedYogi, FedGKT on:
  - [ ] CIFAR-10 IID.
  - [ ] CIFAR-10 non-IID.
  - [ ] CIFAR-100 IID.
  - [ ] CIFAR-100 non-IID.
  - [ ] CINIC-10 IID.
  - [ ] CINIC-10 non-IID.
  - [ ] HAM10000.
- [ ] Use ResNet-56.
- [ ] Use ResNet-110.
- [ ] Report seconds to target accuracy.

### 17.2 Scalability Reproduction

- [ ] Test 20 clients.
- [ ] Test 50 clients.
- [ ] Test 100 clients.
- [ ] Test 200 clients.
- [ ] Use 10% client sampling per round.
- [ ] Target 80% CIFAR-10 IID accuracy using ResNet-110.

### 17.3 Tier Sensitivity Reproduction

- [ ] Evaluate `M = 1` to `M = 7`.
- [ ] Use CIFAR-10 IID.
- [ ] Use 10 clients.
- [ ] Use ResNet-110.
- [ ] Switch profiles every 20 rounds.
- [ ] Report training time to 80% accuracy.

### 17.4 Privacy Reproduction

- [ ] Use CIFAR-10.
- [ ] Use ResNet-56.
- [ ] Use 20 clients.
- [ ] Evaluate distance correlation at `α = 0.00, 0.25, 0.50, 0.75`.
- [ ] Evaluate patch shuffling.
- [ ] Report accuracy.

---

## 18. Key Findings and Conclusions

1. **DTFL significantly speeds up FL** in heterogeneous environments by dynamically assigning clients to tiers.
2. **Client/server split depth matters**: lower tiers reduce client computation but increase communication; higher tiers reduce communication but increase client computation.
3. **Dynamic scheduling is essential** because client resources can change over time.
4. **Local-loss-based training avoids classic split learning synchronization**, allowing client and server to train in parallel.
5. **DTFL maintains comparable accuracy** to state-of-the-art baselines while achieving much lower training time.
6. **More tiers usually reduce training time**, but arbitrary model splitting can harm accuracy; tier design must respect model architecture.
7. **DTFL scales well** to larger numbers of clients under sampled participation.
8. **Privacy techniques can be integrated**, especially distance correlation and patch shuffling, with manageable accuracy trade-offs.
9. **Convergence is theoretically supported** for convex and non-convex objectives under standard FL/local-loss assumptions.

---

## 19. Practical Risks and Engineering Notes

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Poor tier split design | Arbitrary split may reduce accuracy | Split only at module boundaries; validate accuracy per split |
| Communication bottleneck in low tiers | More offloading means larger intermediate transfer | Include communication time in scheduler |
| Server overload | Many clients offloading heavy server-side computation | Monitor server-side time and GPU utilization |
| Stale profiling estimates | Client resources change dynamically | Use EMA and frequent measurement |
| Privacy leakage from feature maps | Intermediate representations can leak information | Add distance correlation, patch shuffling, DP, or separate servers |
| Unbalanced tiers | Some tiers may get too few clients, affecting convergence | Track `A_m`; consider scheduler constraints if necessary |
| Non-IID instability | Label skew can slow convergence | Use fixed Dirichlet split for reproducibility; log per-client distribution |

---

## 20. Minimum Viable Implementation Scope

To implement a thesis/project version of DTFL, prioritize:

- [ ] ResNet module splitting.
- [ ] 7-tier client/server architecture.
- [ ] Local-loss auxiliary heads.
- [ ] Dynamic tier scheduler with EMA.
- [ ] CIFAR-10 IID and non-IID experiments.
- [ ] FedAvg baseline.
- [ ] Time-to-target-accuracy logging.
- [ ] Optional privacy extension with distance correlation.

---

## 21. Full System Success Criteria

A working DTFL implementation should satisfy:

- [ ] Clients can be assigned to different tiers in the same round.
- [ ] Slower clients can offload more layers to the server.
- [ ] Faster clients can train deeper client-side models.
- [ ] Client and server training run in parallel using local loss.
- [ ] Tier assignment changes dynamically across rounds.
- [ ] The system logs computation, communication, and total training time.
- [ ] The system reaches target accuracy faster than FedAvg under heterogeneous profiles.
- [ ] Non-IID partitioning is reproducible using Dirichlet concentration `0.5`.
- [ ] Privacy extensions can be toggled on/off.

---

## 22. Paper Conclusion Restated for Project Planning

DTFL addresses the challenge of collaboratively training large models in heterogeneous environments by dynamically offloading different portions of the global model to the server. It uses local-loss-based split training to let clients and server update in parallel, reducing computation and communication demands on constrained clients and mitigating stragglers. A dynamic tier scheduler assigns clients to tiers based on estimated training time. Experiments on CIFAR-10, CIFAR-100, CINIC-10, and HAM10000 with ResNet-56 and ResNet-110 show large training-time reductions while maintaining model accuracy compared with strong FL and split-learning baselines.
