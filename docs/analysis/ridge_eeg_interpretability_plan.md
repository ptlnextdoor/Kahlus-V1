# Ridge EEG Interpretability Figure Plan

Context: Amrith asked for **visualizations and analysis of the existing benchmark**, not additional benchmarks. The goal is to sanity-check why the ridge baseline performs well on the current EEG future-state forecasting task.

## Current benchmark object

From `src/neurotwin/data/prepared_tasks.py`, the EEG future-state task is built as:

- split prepared EEG recordings into windows
- for each adjacent pair of windows:
  - `X = current EEG window`
  - `Y = next EEG window`
- ridge regression receives flattened `X` rows and predicts flattened `Y` rows
- training uses train-only centering/scaling in `NumpyRidgeBaseline`

So the mentor-facing story should be:

> Ridge is not decoding abstract neural state. It is learning a regularized linear map from recent EEG samples/channels to the immediately following EEG samples/channels. If neighboring windows are autocorrelated, ridge can perform surprisingly well.

## Figure set to generate

### Figure 1. Raw EEG window to ridge input/target

Purpose: show exactly what goes into ridge and what comes out.

Panels:
1. Raw EEG traces for 4-8 channels across one example window pair.
2. Shaded `X/current window` and `Y/next window` regions.
3. Small matrix view showing `X` flattened from `[time, channel]` into a feature vector.
4. Label: `ridge predicts next-window EEG samples, not class labels or new benchmarks`.

What it answers:
- “What actually goes into ridge regression?”
- “What is being predicted?”

### Figure 2. Ridge prediction overlay

Purpose: sanity-check whether ridge is copying smooth temporal structure or capturing meaningful dynamics.

Panels:
1. Actual future EEG traces for selected channels.
2. Ridge-predicted future traces overlaid.
3. Residual trace below each channel.
4. Per-channel Pearson r / MSE annotations.

What it answers:
- “Where is ridge good?”
- “Is it mostly tracking low-frequency/autocorrelated structure?”

### Figure 3. Autocorrelation and lag structure

Purpose: explain why a linear model can do well.

Panels:
1. Autocorrelation curve for representative channels.
2. Cross-window correlation heatmap: current-window channel/time summaries vs future-window channel/time summaries.
3. Mark the forecast horizon/window boundary.

What it answers:
- “Does the task contain short-horizon temporal continuity that favors ridge?”

### Figure 4. Ridge coefficient map

Purpose: inspect learned linear weights.

Panels:
1. Heatmap of coefficient magnitude aggregated by input channel and output channel.
2. Optional time-lag heatmap if coefficient tensor can be reshaped as `[input_time, input_channel, output_time, output_channel]`.
3. Highlight diagonal/channel-local structure if present.

What it answers:
- “Is ridge using same-channel temporal persistence, cross-channel mixing, or broad global leakage-like signals?”

### Figure 5. Split/leakage sanity diagram

Purpose: reassure that this is analysis of the current result, not accidental leakage.

Panels:
1. Train/val/test split schematic at subject/session/run level.
2. Explicit note: normalization fit on train only.
3. Boundary buffer note if used or recommended.

What it answers:
- “Could ridge be winning because of leakage or duplicated adjacent windows across split boundaries?”

## Suggested caption language

> Example diagnostic visualization for the existing MOABB EEG future-state benchmark. The ridge baseline receives a flattened current EEG window `X_t` and predicts the next EEG window `X_{t+1}`. Because the target is a near-future EEG segment, strong short-horizon autocorrelation can make a regularized linear model competitive. These figures are intended as sanity checks of the current benchmark result rather than new benchmark evidence.

## Implementation notes

Use `scripts/analysis/plot_ridge_eeg_diagnostics.py`.

Required real-data input should eventually be one `.npz` containing:

- `x_train`: `[n_train, time, channels]` or `[n_train, features]`
- `y_train`: `[n_train, time, channels]` or `[n_train, features]`
- `x_test`: `[n_test, time, channels]` or `[n_test, features]`
- `y_test`: `[n_test, time, channels]` or `[n_test, features]`
- optional `sfreq`, `channel_names`

If only flattened features are available, pass `--time-length` and `--n-channels` so the script can reshape for waveform figures.
