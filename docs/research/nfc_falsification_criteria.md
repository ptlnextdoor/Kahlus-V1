# NFC Falsification Criteria

NFC must be allowed to fail. These criteria block model-superiority language.

- Old 5-step NFC synthetic signals produced before the corrected gate are
  invalidated as evidence. They may be kept only as debugging history.
- If NFC cannot beat direct baselines on synthetic latent-field recovery, stop
  before A100.
- If no-pair NFC equals full NFC, the pair kernel is decorative.
- If observation-operator-free NFC equals full NFC, compiler factorization is
  decorative.
- If uncertainty does not correlate with error, confidence maps are decorative.
- If any task pads, broadcasts, or silently reshapes predictions to hide shape
  mismatch, the gate fails.
- If an autoregressive or retrieval baseline uses held-out test targets, the
  gate fails.
- If required metrics, ablation rows, or finite uncertainty diagnostics are
  missing, the gate fails.
- If ridge beats NFC on all real fMRI tasks, make no model-superiority claim.
- If real stimulus hash is not verified, make no stimulus-to-fMRI claim.
- If Brain-OF-style or exact Brain-OF matches NFC on leakage-proof translation
  and adaptation tasks, do not claim a generic multimodal advantage.

Synthetic and debug results are plumbing evidence only.

The next A100 step is strict 1x NFC synthetic diagnostic only:

```bash
PYTHONPATH=src python3 -m neurotwin.cli eval \
  --suite nfc_synthetic \
  --out-dir "$RUN_ROOT" \
  --train-steps 50 \
  --seeds 0 1 2 \
  --require-pass
```
