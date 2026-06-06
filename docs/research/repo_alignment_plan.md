# Repo Alignment Plan

## Current State

The repository is the `neurotwin` Python research package and CLI. It already contains the leakage-controlled Neural Translation infrastructure, prepared-manifest training and evaluation, claim gates, model cards, A100 handoff packaging, the legacy `NeuralStateSpaceTranslator`, the historical `NeuroTwinPairOperator`, and the experimental NFC implementation under `src/neurotwin/models/nfc/`.

The current branch is `ptlnextdoor/kahlus-v2-prs` at `cdc3ee9`, with a clean worktree at the start of this pass. The graph report identifies the core abstractions as the old translator, Pair-Operator, split/leakage machinery, prepared tasks, baseline runners, architecture registry, and the new NFC synthetic suite.

## Trusted

- Split manifests, event manifests, leakage audits, prepared-task generation, and claim gates are trusted infrastructure lanes.
- Synthetic and MOABB smoke paths are trusted as plumbing checks, not scientific evidence.
- A100 packaging and handoff scripts are trusted only when built from a clean committed HEAD and verified by checksums.
- The current claim boundary is trusted: no model-superiority, clinical, first-foundation-model, exact TRIBE, exact BrainVista, or fNIRS support claim is allowed.

## Not Trusted Yet

- NFC synthetic results are not scientific evidence.
- The NFC synthetic falsification gate is the immediate trust boundary before Algonauts/CNeuroMod or 6x A100 runs.
- The benchmark layer still needs scrutiny for over-broad labels, ablation thresholds, uncertainty calibration semantics, and whether the gate proves more than plumbing.
- Old short-step synthetic signals are invalidated as evidence after the corrected gate.
- fNIRS and TurboQuant/TurboVec are research notes or optional future infrastructure, not implemented model claims.

## Planned Changes

- Add/update the top-level NFC pivot project-state document.
- Add/update the NFC mathematical constitution and no-loss equation ledger.
- Add/update a code architecture map tying research concepts to files and claim boundaries.
- Add/update a master research-state dossier.
- Add fNIRS observation-operator notes as docs only.
- Add TurboQuant/TurboVec retrieval/audit notes as docs only.
- Tighten claim hygiene docs.
- Verify and, only if necessary, harden the NFC synthetic gate and A100 handoff strictness.

## Out of Scope

- No A100 jobs.
- No raw public neural data.
- No clinical, MDD, diagnostic, or treatment claims.
- No full fNIRS implementation.
- No TurboVec dependency or vendored retrieval stack.
- No deletion of old translator, Pair-Operator, MOABB, leakage, reporting, model-card, or packaging systems.
- No claim that NFC beats baselines unless the strict evidence gate actually passes.

## Validation Plan

- `git diff --check`
- `PYTHONPATH=src python3 -m unittest discover -s tests -v`
- `PYTHONPATH=src python3 -m neurotwin.cli doctor`
- `PYTHONPATH=src python3 -m neurotwin.cli eval --suite nfc_synthetic --out-dir /tmp/neurotwin_nfc_alignment_smoke --train-steps 1 --seed 0`
- `PYTHONPATH=src python3 -m neurotwin.cli eval --suite nfc_synthetic --out-dir /tmp/neurotwin_nfc_alignment_strict --train-steps 5 --seed 0 --require-pass || true`
- `bash scripts/run_smoke.sh /tmp/neurotwin_alignment_smoke`
- `graphify update .`
- Package only after validation and only from a clean committed HEAD if a commit is made.
