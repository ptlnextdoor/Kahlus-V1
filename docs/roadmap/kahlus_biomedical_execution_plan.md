# Kahlus Biomedical Execution Plan

## RewardDock Clinical Extension Strategy

Existing at-home and point-of-care saliva workflows already cover cortisol and salivary
alpha-amylase collection/readout patterns. RewardDock should not compete as a generic hormone or
stress meter; those biosensors are ingredients for a task-specific response-profile product.

Kahlus RewardDock Clinical =

- standardized task battery
- webcam pupillometry
- reaction time / effort behavior
- optional PPG/HRV
- optional cortisol / alpha-amylase module
- Kahlus response-profile model
- clinician dashboard

Primary clinical wedge:

Adults with anhedonia-related depression or social anxiety who are starting or adjusting treatment.

Need statement:

A way to objectively track reward-response and stress-recovery changes in adults undergoing
treatment for anhedonia-related depression or social anxiety in order to help clinicians identify
ineffective treatment plans earlier than self-report alone.

The goal is not to replace interviews or self-report. The goal is to add repeated within-person
response data under standardized conditions and ask whether a person's reward/stress/social-response
profile is improving relative to their own baseline. It does not ask whether the person has
depression, anhedonia, or social anxiety.

Allowed claims:

- tracks reward-response profile
- tracks stress-recovery profile
- tracks within-person change over time
- supports treatment-response research
- supports clinician review as an adjunct signal

Blocked claims:

- diagnoses depression
- diagnoses anhedonia
- diagnoses social anxiety
- recommends medication
- adjusts medication
- treats depression
- treats PTSD
- replaces clinician judgment
- claims cortisol/alpha-amylase alone measures anhedonia

## RewardDock Biomarker Architecture

Core v0:

- webcam pupil dilation
- reaction time
- effort persistence
- task accuracy
- self-report slider

Core v1:

- PPG/HRV
- heart-rate recovery
- optional EDA

Future biochemical add-on:

- salivary cortisol
- salivary alpha-amylase

Avoid as primary:

- dopamine
- oxytocin
- epinephrine/norepinephrine direct measurement

Dopamine, oxytocin, and epinephrine are not practical home biomarkers for this use case because
peripheral levels do not cleanly map to central reward circuitry, assays are difficult or
context-dependent, and they are not condition-specific enough for RewardDock v0. Cortisol and
salivary alpha-amylase are more realistic stress-context add-ons, but they should not be the core
product.

## RewardDock Device Roadmap

RewardDock v0:

- webcam pupil + reward task + reaction time + self-report

RewardDock v1:

- add PPG/HRV

RewardDock v2:

- add optional saliva cortisol / alpha-amylase integration

RewardDock v3:

- clinician dashboard + longitudinal treatment-response report

RewardDock Consumer:

- future nonclinical self-improvement product only after clinical/research validation

Do not start with self-improvement branding. Start clinical/research first to avoid becoming a
generic dopamine-maxxing wellness app.

## RD-0: Safe Scaffold

Deliverables:

- root technical roadmap docs
- ResearchDock dataclasses
- safe task templates
- deterministic synthetic sessions
- numpy-only metrics
- synthetic data card
- branch-specific evidence gate
- local smoke script

Acceptance:

- no frozen v1 load-bearing files modified
- no A100 or cluster jobs
- no hardware access
- no real participant data
- no clinical claims
- tests pass

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_synth
git diff --check -- ':!graphify-out'
```

## RD-1: Task App Prototype

- task flow for reward anticipation, effort-for-reward, mild frustration, recovery, and visual attention
- webcam/pupil interface design
- optional PPG/HRV input schema
- CSV/session export
- quality flags

No clinical claims and no treatment tasks.

RD-1 local implementation scope:

- deterministic task/session protocol in code
- design-only webcam pupil/gaze and optional PPG/HRV interface contract
- no camera or PPG device opened in tests or scripts
- CSV export for sessions, trials, sensor packets, task events, and self-report
- quality-flag summaries for missing pupil, invalid pupil, implausible reaction time, and invalid accuracy

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd1 --seed 0 --write-session-export
```

## RD-2: Kahlus v2 Multimodal Observation Model

- synthetic pupil + behavior + PPG pretraining
- observation operators for pupil, HRV, behavior, and EEG
- subject adapter design
- missing-modality tests

Baselines must run before any Kahlus model comparison.

RD-2 local implementation scope:

- synthetic ResearchDock multimodal observation task with subject-held-out sessions
- behavior/task/self-report features predict pupil and HRV proxy targets
- mean and ridge baselines run before the ResearchDock observation-operator candidate
- NumPy low-rank residual observation operator with a simple subject-adapter offset
- synthetic pretraining artifact writer; no real data, no EEG hardware, and no clinical claims

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd2 --seed 0 --run-observation-model
```

## RD-3: Public Dataset Ingestion Review

- WESAD feature mapping
- DEAP feature mapping
- SEED feature mapping
- data terms and citation verification
- no invented URLs or loaders

RD-3 local implementation scope:

- source-backed review registry for WESAD, DEAP, and SEED
- ResearchDock field mappings for physiology, EEG, eye-tracking, self-report, and stimulus context
- explicit access and license uncertainty where terms are not fully resolved
- JSON and Markdown review artifacts
- no dataset downloads, no executable loaders, no raw participant data, and no clinical claims

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd3 --seed 0 --write-public-dataset-review
```

## RD-4: Pilot Readiness Preflight

- validation-scale pilot manifest
- required prior evidence checklist
- pre-collection safety gate
- operator checklist for storage, identifier handling, and artifact archival
- no legal advice, no real participant data, no hardware access, no collection, and no clinical claims

RD-4 local implementation scope:

- build a JSON-compatible ResearchDock pilot manifest from synthetic fixtures and the RD-1 protocol
- require RD-0 synthetic gate, RD-1 export contract, RD-2 baseline artifacts, and RD-3 dataset review before any pilot path
- block hardware access, real participant data, PII, clinical labels, stimulation, trauma exposure tasks, and diagnostic/treatment claims
- write manifest, preflight gate, and Markdown report artifacts

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd4 --seed 0 --write-pilot-preflight
```

## RD-5: Response-Profile Readiness Audit

- future latent response-profile clustering is audited for data readiness only
- no clustering is performed, no cluster labels are emitted, and no clinical claims are supported
- the audit checks profile-vector finiteness, minimum validation-scale session count, observed quality flags, and missing-pupil blockers
- synthetic RD-5 output is expected to fail readiness honestly because it is tiny and includes a missing-pupil profile

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd5 --seed 0 --write-profile-readiness
```

## RD-6: Observation Missing-Modality Audit

- RD-2 observation-task construction now reports how many synthetic trials are eligible versus skipped
- skipped trials are counted by missing sensor packet, pupil diameter, HRV proxy, and behavior-response fields
- the observation task JSON, metrics JSON, and Markdown report expose the audit instead of silently dropping missing-modality rows
- no imputation, clustering, hardware access, real participant data, or clinical claims are introduced

Verification:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests/researchdock -v
PYTHONPATH=src python3 scripts/run_researchdock_synthetic.py --out-dir /tmp/kahlus_researchdock_rd6 --seed 0 --run-observation-model
```

## A100-0: 7xA100 Handoff Readiness Package

- exact commit hash
- clean worktree proof
- runner tarball
- config files
- checksum manifest
- CPU smoke command
- DDP/torchrun command
- honest GPU count labeling: 7xA100, not 8xA100
- evidence bundle writer and returned-evidence audit command
- no secrets, checkpoints, raw arrays, or raw private participant data

A100-0 local implementation scope:

- package only from a clean git worktree; dirty worktrees fail instead of fabricating proof
- write `COMMIT_HASH.txt`, `CLEAN_WORKTREE.txt`, `A100_HANDOFF_MANIFEST.json`, `README_A100_7X_HANDOFF.md`, `SHA256SUMS`, and a runner tarball
- set `--nproc_per_node=7` and `--expected-gpus 7`
- do not launch A100, Slurm, Docker, torchrun, dataset downloads, or full sweeps
- keep claims at infrastructure handoff only

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-1: 7xA100 Returned-Evidence Audit Default

- default returned-evidence audit expectation is 7 visible GPUs
- 7-GPU synthetic evidence passes without requiring an explicit override
- 8-GPU evidence fails by default unless the caller intentionally overrides `--expected-gpus`
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_audit_ktm_a100_evidence -v
```

## A100-2: Runner Self-Smoke Importability

- the 7xA100 runner tarball includes the script bootstrap helper needed by packaged CLI scripts
- the packaged returned-evidence audit script can print `--help` from inside an extracted runner
- the runner still excludes secrets, checkpoints, raw arrays, raw private participant data, and graph outputs
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-3: Runner Evidence-Bundle Self-Smoke

- the extracted 7xA100 runner can package a tiny returned-evidence zip using its included evidence writer
- if no handoff README template is available inside the runner, the writer emits a minimal claim-safe README instead of failing
- the generated evidence README keeps boundaries explicit: no scientific result, no clinical claim, no diagnosis, no treatment, no recovery claim, and no model-superiority claim
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-4: Runner Fail-Closed Audit Round Trip

- the extracted 7xA100 runner can audit the tiny self-smoke evidence zip it creates
- incomplete evidence fails closed with audit JSON and Markdown report artifacts instead of crashing
- missing required run files are reported with the canonical `required_file_missing` finding code
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-5: Operator Runner Self-Smoke Command

- the 7xA100 handoff manifest and README include a single local `runner_self_smoke_command`
- the extracted runner includes `scripts/smoke_a100_runner.py`
- the self-smoke creates a tiny evidence zip, audits it, verifies fail-closed report artifacts, and prints `runner_self_smoke_passed=true`
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-6: Runner Internal Checksum Command

- the 7xA100 handoff manifest and README include `runner_checksum_command`
- after extracting the runner tarball, `shasum -a 256 -c RUNNER_SHA256SUMS` verifies runner contents before self-smoke
- no A100, Slurm, Docker, torchrun, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-7: Extracted Runner CPU Smoke

- the advertised `cpu_smoke_command` is executed from inside the extracted 7xA100 runner during tests
- the runner includes the minimal `neurotwin` package initializer and unified gate package needed by ResearchDock tests
- this verifies the ResearchDock CPU smoke path locally before any A100, Slurm, Docker, torchrun, dataset download, or returned real evidence

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-8: Advertised CLI Train Target Import Smoke

- the extracted 7xA100 runner includes `neurotwin.cli` and the local command/training dependency surface needed by the advertised DDP command target
- `PYTHONPATH=src python3 -m neurotwin.cli train --help` runs inside the extracted runner
- this validates command-target importability only; it does not launch A100, Slurm, Docker, torchrun, training, dataset download, or returned real evidence

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-9: Advertised Train Command Parser Alignment

- the 7xA100 DDP command no longer advertises unsupported `train --require-pass`
- the extracted runner dry-runs `PYTHONPATH=src python3 -m neurotwin.cli train --dry-run --config configs/train/moabb_a100_smoke.yaml`
- this validates command/config parseability only; it does not launch A100, Slurm, Docker, torchrun, training, dataset download, or returned real evidence

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-10: Runner OS Metadata Exclusion

- tracked `.DS_Store` / macOS metadata files are excluded from the handoff and runner tarball
- the handoff test injects a tracked `.DS_Store` into a copied source tree and verifies it is not shipped
- no A100, Slurm, Docker, torchrun, training, dataset download, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-11: Runner Model Artifact Exclusion

- tracked serialized model/baseline artifacts such as `.safetensors`, `.onnx`, `.pkl`, `.pickle`, and `.joblib` are excluded from the handoff and runner tarball
- the handoff test injects tracked model artifact fixtures into included source paths and verifies they are not shipped
- no A100, Slurm, Docker, torchrun, training, dataset download, checkpoint transfer, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## A100-12: Runner Source Symlink Refusal

- tracked symlinks inside runner source paths are refused before copying so the package cannot dereference outside files into the runner
- the handoff test injects a tracked symlink under an included source directory and verifies packaging fails closed with a symlink error
- no A100, Slurm, Docker, torchrun, training, dataset download, checkpoint transfer, or returned real evidence is required

Verification:

```bash
PYTHONPATH=src python3 -m unittest tests.artifacts.test_kahlus_a100_7x_handoff -v
```

## Long-Term

- validation-scale ResearchDock data collection after review
- v3 perturbation-response training when structured data exists
- A100 handoff only after local correctness, split audit, baseline ladder, evidence gate, clean merge/tag, smoke run, and returned evidence audit
