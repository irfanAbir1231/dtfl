# Privacy Implementation Plan

This plan maps the thesis privacy guidance to the current DTFL codebase. It describes where to change and how, without implementing code.

## Goals from Thesis
- Apply privacy on the feature (smashed) tensor, not raw inputs.
- Use norm clipping and Gaussian noise.
- Use an adaptive noise decay schedule (higher noise early, lower later).
- Track privacy metrics: SNR and correlation drop.
- Optionally report (epsilon, delta) via a DP accountant.

## Current Data Flow (Baseline)
- Client produces features (smashed tensor) and sends to server.
- Server consumes features, computes loss, and backpropagates gradients.
- Client trains with local loss and optional decorrelation loss.

Relevant locations:
- Client forward and feature send: main.py (Client.train)
- Server training: main.py (train_server)
- Feature tensor definition: model/resnet.py (ResNet.forward)
- Distance correlation utility: utils/loss.py (dis_corr)

## Change Points and How

### 1) Add Privacy CLI Arguments
Where: main.py (argparse block)

Add args to configure the privacy mechanism and metrics:
- privacy_enable (bool)
- clip_C (float)
- noise_sigma0 (float)
- noise_sigma_min (float)
- noise_decay (float, e.g., 0.9 to 0.99)
- privacy_delta (float)
- privacy_metrics (bool)
- privacy_log_interval (int)

Rationale:
- These parameters are required to implement norm clipping, adaptive noise, and metric logging.

### 2) Noise Schedule (Adaptive Decay)
Where: main.py (global training loop over rounds)

Use the round index to compute sigma_t, for example:
- sigma_t = max(sigma_min, sigma0 * (decay ** round))

Pass sigma_t to Client.train or store it in a global variable.

Rationale:
- The thesis recommends higher noise early and lower noise later.

### 3) Privacy Mechanism on Smashed Tensor
Where: main.py (Client.train, right after extracted_features, fx)

Replace the current plain send with:
1) Compute or retrieve clip bound C.
2) Clip fx to L2 norm C.
3) Add Gaussian noise with std = sigma_t * C.
4) Send the noisy tensor to train_server.

Rationale:
- This is the exact point where smashed data is produced and sent.

### 4) Privacy Metrics: SNR
Where: main.py (Client.train, same block as noise injection)

Compute SNR between the clipped signal and injected noise:
- signal = fx_clipped
- noise = fx_noisy - fx_clipped
- SNR = E[||signal||^2] / E[||noise||^2]

Log SNR to W&B (per batch or per epoch).

Rationale:
- This is explicitly required in the thesis privacy metrics section.

### 5) Privacy Metrics: Correlation Drop
Where: main.py (Client.train, same block as noise injection)

Use existing dis_corr:
- dcor_clean = dis_corr(images, fx_clipped)
- dcor_noisy = dis_corr(images, fx_noisy)
- corr_drop = (dcor_clean - dcor_noisy) / max(dcor_clean, eps)

Log corr_drop to W&B.

Rationale:
- Correlation drop measures how much dependence was removed.

### 6) Optional DP Accountant for (epsilon, delta)
Where: main.py (global training loop)

If strict privacy accounting is needed, add a DP accountant using:
- sampling rate q = batch_size / local_dataset_size
- noise multiplier sigma_t
- number of steps per round

Log epsilon over time.

Rationale:
- The thesis mentions (epsilon, delta) guarantees. This step adds formal accounting.

## Notes on Server Side
- Server consumes the privacy-protected features and does not require changes.
- No changes to train_server are needed beyond optional logging.

## Suggested File Organization (Optional)
If you want to keep main.py clean, add a small privacy helper module:
- utils/privacy.py
  - clip_features
  - add_gaussian_noise
  - compute_snr
  - compute_corr_drop
  - noise_schedule

Then import and use these functions in main.py.

## Acceptance Checklist
- Privacy args are configurable from CLI.
- Adaptive sigma schedule is applied per round.
- Smashed tensor is clipped and noised before server send.
- SNR and correlation drop are logged each round.
- (Optional) epsilon tracking is reported.
