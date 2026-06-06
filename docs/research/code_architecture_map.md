# Code Architecture Map

This map ties research concepts to current repository files. Treat it as a navigation layer, not a replacement for tests or code review.

## Research Concepts to Files

| Research concept | Current files | Status |
| --- | --- | --- |
| NFC core | `src/neurotwin/models/nfc/` | implemented experimental |
| Neural Field Compiler config/model | `src/neurotwin/models/nfc/compiler.py` | implemented experimental |
| Latent neural field | `src/neurotwin/models/nfc/latent_field.py` | implemented experimental |
| Causal field update | `src/neurotwin/models/nfc/field_update.py` | implemented experimental |
| Low-rank pair kernel | `src/neurotwin/models/nfc/pair_kernel.py` | implemented ablation primitive |
| Observation operators | `src/neurotwin/models/nfc/observations/` | implemented fMRI/EEG/behavior operators |
| Synthetic field generator | `src/neurotwin/data/synthetic_field.py` | implemented, gate pending |
| NFC falsification suite | `src/neurotwin/benchmarks/nfc_suite.py` | critical gate |
| Architecture registry | `src/neurotwin/models/architecture_registry.py` | canonical model dispatch |
| Old translator | `src/neurotwin/models/torch_models.py` | baseline/current model |
| Pair-Operator | `src/neurotwin/models/pair_operator.py` | historical ablation |
| Local baselines | `src/neurotwin/models/baselines.py`, `src/neurotwin/benchmarks/baseline_suite.py` | implemented |
| Prepared tasks | `src/neurotwin/data/prepared_tasks.py` | trusted task layer |
| Leakage audits | `src/neurotwin/data/audit.py`, `src/neurotwin/data/leakage.py`, `src/neurotwin/eval/audit.py` | trusted infrastructure |
| Split manifests | `src/neurotwin/data/split_manifest.py`, `src/neurotwin/data/manifest_io.py` | trusted infrastructure |
| Evidence gates | `src/neurotwin/reports/evidence_gate.py`, `src/neurotwin/eval/paper_gate.py` | trusted if finalization is explicit |
| Model cards/reports | `src/neurotwin/reports/`, `src/neurotwin/benchmarks/reports.py` | read-only reporting path |
| CLI | `src/neurotwin/cli.py`, `src/neurotwin/eval/command.py`, `src/neurotwin/training/command.py` | public command surface |
| Runtime estimates/preflight | `src/neurotwin/runtime/` | cluster safety layer |
| A100 handoff | `scripts/a100_krish_agent_autorun.sh.in`, `scripts/package_runner_bundle.sh`, `scripts/package_a100_handoff_zip.sh`, `README_HANDOFF.md.in` | package after clean commit |
| TurboVec layer | `src/neurotwin/retrieval/` if added | optional/deferred |
| fNIRS | docs only | theory/deferred |

## Files Not to Mutate Casually

- `src/neurotwin/data/split_manifest.py`
- `src/neurotwin/data/audit.py`
- `src/neurotwin/eval/audit.py`
- `src/neurotwin/reports/evidence_gate.py`
- `src/neurotwin/eval/paper_gate.py`
- `src/neurotwin/training/prepared_loop.py`
- `scripts/package_runner_bundle.sh`
- `scripts/package_a100_handoff_zip.sh`
- `README_HANDOFF.md.in`

These files define trust boundaries, packaging behavior, or long-running experiment contracts.

## Claim Boundary Files

- `README.md`
- `docs/CLAIMS.md`
- `docs/BASELINES.md`
- `docs/ROADMAP.md`
- `docs/research/neurotwin_project_state.md`
- `docs/research/nfc_falsification_criteria.md`
- `docs/research/neurotwin_master_research_state.md`

Any change to these files must preserve the rule that synthetic and smoke results are plumbing checks unless evidence gates explicitly permit a claim.

## Current Falsification Gate Files

- `src/neurotwin/benchmarks/nfc_suite.py`
- `src/neurotwin/eval/command.py`
- `src/neurotwin/cli.py`
- `tests/benchmarks/test_nfc_suite.py`
- `tests/cli/test_expanded.py`
- `docs/research/nfc_falsification_criteria.md`

The gate must fail hard on shape mismatch, missing metrics, NaNs, target leakage, and non-pass status under `--require-pass`.

## Files to Check Before A100 Handoff

- `git status --short`
- `src/neurotwin/benchmarks/nfc_suite.py`
- `src/neurotwin/eval/command.py`
- `scripts/a100_krish_agent_autorun.sh.in`
- `README_HANDOFF.md.in`
- `scripts/package_runner_bundle.sh`
- `scripts/package_a100_handoff_zip.sh`
- `docs/research/neurotwin_project_state.md`
- `docs/research/nfc_falsification_criteria.md`

The next A100 step is strict 1x NFC synthetic diagnostic only. Do not jump to Algonauts or 6x DDP before the synthetic gate passes.
