# BNCI2014_001 ridge sanity analysis: what the baseline actually predicts

**Scope.** Sanity analysis of an existing public motor-imagery EEG benchmark
(BNCI2014_001 / BCI Competition IV-2a, 22 EEG channels, fs = 250 Hz), pulled
from the local MOABB cache. Subject-held-out split: train subjects 1-6, test
subjects 7-9, per-subject z-scoring (no cross-subject normalization leakage).
No clinical, diagnostic, or foundation-model claim is made here.

## What ridge receives and predicts

The repo's `linear_ridge` baseline (`NumpyRidgeBaseline`, alpha=1e-2) reshapes
each `[windows, time, channels]` batch into `[windows*time, channels]` and fits
a **channel-to-channel** linear map. It does not see the time axis as features;
it learns a regularized 22x22 mixing matrix. The forecasting task builds windows
as `X = signal[s : s+L]` and `Y = signal[s+h : s+h+L]` with `L = 128` samples
(512 ms) and horizon `h`.

## The headline number is short-horizon continuity

At the historical `horizon = 1` setting, the target window overlaps the input
window in **127 of 128 samples**. A near-identity channel map therefore scores
high almost by construction:

| horizon | overlap | ridge Pearson r | ridge R^2 | persistence r |
|--------:|--------:|----------------:|----------:|--------------:|
| 1 (4 ms)   | 0.99 | **0.866** | 0.749 | 0.852 |
| 4 (16 ms)  | 0.97 | 0.599 | 0.358 | 0.565 |
| 16 (64 ms) | 0.88 | 0.243 | 0.051 | 0.142 |
| 64 (256 ms)| 0.50 | 0.008 | -0.005 | -0.022 |
| 128 (512 ms)| 0.00 | 0.056 | 0.002 | -0.074 |
| 192 (768 ms)| 0.00 | 0.098 | 0.009 | -0.079 |

Ridge tracks the input/target overlap fraction and collapses to the noise floor
(r ~ 0.05-0.10, R^2 ~ 0) once the predicted future no longer overlaps the
observed window. It stays only marginally above matched persistence throughout,
which confirms the gain is waveform continuity, not learned neural dynamics.

## What the figure shows

`figures/fig_ridge_overlap_headline.pdf` is a single 4-panel figure:

- **(a)** raw input vs target for one electrode: the target trace is drawn
  directly on top of the input, offset by `h` samples, so the copy is visible
  by construction rather than asserted in prose.
- **(b)** test Pearson r for ridge and matched persistence vs. horizon
  (log-x): both collapse to the noise floor as `h` grows, and ridge never
  separates far from persistence.
- **(c)** the central result: r plotted directly against overlap fraction
  (not against `h`), on one axis. Skill tracks overlap monotonically down to
  the noise floor.
- **(d)** the fitted 22x22 ridge coefficient map. It is **diagonally
  dominant, not an identity**: mean diagonal beta = 0.79 (5.1x the mean
  |off-diagonal| entry), but `||C-I||_F / ||I||_F = 0.94` -- 80% of
  coefficient mass is off-diagonal. The strong diagonal is why ridge tracks
  persistence; the off-diagonal mass is the small cross-channel term that
  lifts it slightly above persistence at short horizons.

The LaTeX caption ships alongside the figure at
`figures/fig_ridge_overlap_headline_caption.tex`.

## What stays unexplained / honest caveats

- The cross-channel covariance term (panel d's off-diagonal mass) is real,
  now characterized as 20% of |beta| mass at 5x lower average magnitude than
  the diagonal, but *why* those specific channel pairs carry it is not yet
  investigated.
- These results are one split and one z-scoring choice; I have not swept alpha,
  window length, or per-subject variance.
- I have not yet run the Kahlus model on this exact overlap-free (h >= L) task,
  which is the only setting where beating the noise floor would be meaningful.
- No spectral or phase-based baseline is included yet.

## Takeaway

The strong ridge number is a **task-construction artifact of overlapping
input/target windows**, not evidence of a rich neural-state model. The honest
benchmark is the non-overlapping horizon (h >= L), where every current baseline
is at the noise floor. That is the bar any real model, including Kahlus, has to
clear.
