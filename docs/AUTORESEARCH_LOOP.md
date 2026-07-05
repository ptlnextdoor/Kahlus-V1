# Kahlus Recursive Autoresearch Loop

This is the standing operating protocol for agents that recursively improve
Kahlus-V1. It does not replace `AGENTS.md`; the Kahlus constitution and claim
boundaries remain authoritative unless Krish explicitly overrides them in the
current task.

## End Goal

Build a repeatable agent loop that keeps shipping small, evidence-gated features
toward Kahlus as a leakage-controlled neural forecastability and neural
translation engine. The loop should make the final research target more true:

- leakage-proof Neural-CASP gates and artifact contracts;
- residual forecastability under held-out subject/site/dataset splits;
- missing-modality reconstruction and cross-modal neural translation;
- future-state forecasting and few-shot subject adaptation;
- NFC / Neural Field Compiler falsification and recovery gates.

The loop must not drift into unsupported claims. Kahlus is not:

- a clinical product;
- a seizure predictor;
- a depression classifier;
- a brain foundation model;
- a proven model-superiority result.

It also must not claim:

- NeuroTwin beats baselines;
- NFC is proven;
- NeurIPS-quality result;
- clinical digital twin;
- diagnostic model;
- treatment predictor;
- exact TRIBE v2 reproduction;
- exact BrainVista reproduction;
- A100 success if no evidence bundle exists.

Scientific claims require real prepared data, leakage audit pass,
validation-selected checkpoints, final held-out test metrics, and an explicit
`scientific_claim_allowed=true` decision in `summary.json`. A passed
`paper_mode_gate.json` is an artifact-contract result, not automatic scientific
claim permission.

## Required First Reads

Every iteration starts by reading these files from the current `main` branch:

- `AGENTS.md`
- `README.md`
- `docs/CLAIMS.md`
- `docs/ROADMAP.md`
- `docs/research/neurotwin_project_state.md`
- `docs/research/neurotwin-technical-report.md`
- `/Users/krishgarg/Documents/papers/neuroML/cluster/A100_CLUSTER_AGENT_RUNBOOK.md`

If these disagree, follow the stricter claim boundary and stop before launching
any run whose protocol would be mislabeled.

## Loop Contract

One loop iteration is one bounded research-to-code cycle:

1. **Ground:** Read the required docs, current git status, recent result ledgers,
   and any active task brief.
2. **Research:** Gather source evidence for the narrow feature. Use official
   APIs or primary sources where available, not scraped summaries.
3. **Select:** Pick one small feature that advances a roadmap gate and can be
   validated with local tests, a smoke run, or at most four A100 GPUs.
4. **Specify:** Write the claim boundary, expected artifacts, tests, and
   failure conditions before editing.
5. **Implement:** Work on a branch or isolated worktree. Keep changes scoped.
6. **Verify:** Run targeted unit tests and any required smoke/audit gates.
7. **Cluster only if needed:** Use the A100 runbook, dynamically select free
   GPUs, and cap at `MAX_GPUS=4` unless Krish explicitly authorizes more.
8. **Package evidence:** Store logs, metrics, configs, citations, and gate
   decisions. Exclude secrets, checkpoints, raw arrays, tarballs, and nested
   archives from sendable bundles.
9. **Report:** Summarize what changed, what passed, what failed honestly, the
   actual GPU count used, and the next best loop candidate.

No loop may claim success from intent alone. The evidence must prove the exact
claim being made.

Every scientific or benchmark-facing artifact must carry `claim_scope` and
`stop_reason`, then satisfy or explicitly fail the Kahlus gate predicate:

```text
GatePass = C_split ∧ C_finite ∧ C_baseline ∧ C_controls ∧ C_power ∧ C_claim_scope
```

The gate fields are:

- `C_split`: held-out subject/site/dataset/modality split is audited.
- `C_finite`: payloads and metrics contain no NaN/Inf.
- `C_baseline`: the model beats the best relevant baseline, including moving
  average or persistence when applicable.
- `C_controls`: shuffle, time-shift, permutation null, subject probe, and
  synthetic NULL controls behave as expected.
- `C_power`: event count, positive windows, clusters, and bootstrap count are
  adequate for the scoped claim.
- `C_claim_scope`: `claim_scope` and `stop_reason` are present and honest.

Aggregate metrics must never be promoted as whole-model success when sidecar
tasks disagree. Report aggregate and sidecar metrics separately.

## Agent Team

Use fresh, bounded agents when parallel work is useful. Do not make agents share
unbounded context or write overlapping files.

| Role | Responsibility | Output |
| --- | --- | --- |
| Conductor | Holds the `/goal`, chooses the next feature, and resolves tradeoffs. | Iteration brief and final status. |
| Research Scout | Finds primary sources, reviews, baselines, and prior art. | Source table with IDs, URLs, limitations, and claim impact. |
| Kahlus Architect | Maps evidence to Kahlus gates, task IDs, configs, and artifact contracts. | Feature spec and acceptance checks. |
| Implementer | Makes the smallest scoped code/doc change on a branch. | Patch plus local test output. |
| Reviewer | Checks claim hygiene, leakage discipline, tests, and reproducibility. | Approval or concrete required fixes. |
| Cluster Runner | Uses the A100 runbook for GPU work. | Protocol log, selected GPUs, Docker/Slurm logs, evidence zip. |
| Evidence Packager | Builds the sendable bundle and forbidden-file scan. | Zip, checksum, contents list, and exclusion report. |

Suggested concurrency: at most one implementer, one reviewer, one research scout,
and one cluster runner at a time. Multiple implementers may run only with
disjoint file ownership.

## Research Intake

Use the right source for the question:

- PubMed/PMC for biomedical human evidence and PubMed IDs.
- arXiv for preprints, but do not treat preprints as settled medical evidence.
- Semantic Scholar and OpenAlex for citation graph expansion and related work.
- OpenAI Deep Research for synthesis over curated sources, not for bulk crawling.
- Repo-local docs and result ledgers for Kahlus claim boundaries and current
  implementation state.

Good external tooling candidates:

- OpenAI Deep Research supports web search, file search, remote MCP servers, and
  code interpreter for analysis.
- Prefect is the preferred workflow engine if the loop becomes scripted because
  it supports dynamic runtime branching and run monitoring.
- MLflow or W&B can be used as the experiment system of record. Pick one source
  of truth before logging production evidence.

Every literature note must record query text, source, timestamp, identifiers
such as DOI/PMID/PMCID/arXiv/Semantic Scholar/OpenAlex IDs, and the exact claim
that the source does or does not support.

## Exercise And Depression Evidence Gate

This gate applies only to features that touch exercise, depression, mental
health coaching, symptom tracking, or behavioral recommendations. It does not
force exercise/depression research into unrelated neural-model work.

Allowed source priority:

1. Human systematic reviews, meta-analyses, and network meta-analyses.
2. Human randomized controlled trials.
3. Human observational studies only for association or prevention hypotheses.
4. Guidelines or safety-screening sources for risk framing.

Anchor sources to keep in the seed bibliography:

- Clegg et al., Cochrane Database Syst Rev 2026, `PMID:41500513`.
- Noetel et al., BMJ 2024 network meta-analysis, `PMID:38355154`.
- Singh et al., Br J Sports Med 2023 umbrella review, `PMID:36796860`.
- ACSM preparticipation health screening update, `PMID:26473759`.

Allowed wording:

- "may help reduce depressive symptoms"
- "can be considered alongside clinician-directed care"
- "is associated with lower depression symptoms/risk" for observational data

Forbidden wording:

- "cures depression"
- "replaces medication or therapy"
- "diagnoses depression"
- "predicts individual treatment response"
- "clinically validated intervention" unless the exact validation exists

Minimum evidence record:

```json
{
  "citation": "",
  "url": "",
  "pmid": "",
  "doi": "",
  "study_design": "systematic_review | meta_analysis | RCT | observational | guideline",
  "population": "",
  "intervention": "",
  "comparator": "",
  "outcomes": "",
  "risk_of_bias": "low | some_concerns | high | unclear",
  "certainty": "high | moderate | low | very_low",
  "allowed_claim": "",
  "limitations": "",
  "last_verified": "YYYY-MM-DD"
}
```

Stop or require clinician/safety framing for severe symptoms, self-harm risk,
mania/psychosis history, eating disorder or exercise compulsion, pregnancy or
postpartum risk, major chronic illness, medication changes, acute injury, chest
pain, fainting, severe shortness of breath, palpitations, or marked mood
worsening. For U.S. self-harm or crisis signals, point to call/text/chat `988`
or emergency services for imminent danger.

## Feature Selection Rubric

Score candidates before implementation:

| Criterion | Pass condition |
| --- | --- |
| Roadmap fit | Directly advances Neural-CASP, NFC falsification, MOABB evidence, translation, forecasting, or adaptation gates. |
| Evidence fit | Has primary/review evidence or a repo-local contract proving why it matters. |
| Claim hygiene | Cannot create a forbidden clinical, superiority, or foundation-model claim. |
| Blast radius | Small enough for one branch and focused tests. |
| Testability | Has a deterministic local test, dry run, smoke run, or bounded A100 check. |
| Compute discipline | Uses CPU first; if A100 is needed, uses no more than four free GPUs by default. |
| Artifact value | Produces a JSON/Markdown/log artifact useful for the paper ledger. |

Prefer work that hardens gates over work that only makes the model larger.

## First Backlog

Start with these candidates unless the current roadmap supersedes them:

1. Task-ID aliasing between paper task names and executable prepared tasks.
2. Prepared `eeg_meg_to_shared_latent_state` smoke task.
3. Prepared `fmri_to_eeg_meg_spectral_proxy` task.
4. True minute-based few-shot support instead of only support-window counts.
5. Stronger NFC falsification margins.
6. Uncertainty-error correlation as an NFC gate.
7. Architecture-registry coverage for adaptation, generalization, and spectral
   proxy tasks.
8. Weak or pseudo-paired matching by `stimulus_segment_id`.
9. Real-data acceptance report for the "three MOABB reports" roadmap gate.

## A100 Policy

Use the reusable cluster runbook at:

```text
/Users/krishgarg/Documents/papers/neuroML/cluster/A100_CLUSTER_AGENT_RUNBOOK.md
```

Repo-local A100 docs such as `docs/A100_RUNBOOK.md` describe canonical full-run
lanes and may mention 6-GPU or larger jobs. For this recursive autoresearch
loop, the stricter policy here wins: MAX_GPUS=4 by default, truthful `NxA100`
labels, and stop before exact-count protocols that need more than four GPUs
unless Krish explicitly authorizes that protocol change.

Non-negotiable rules:

- Never print, copy, upload, archive, or commit `pw.txt`.
- Probe current cluster tools, disk, GPUs, and active processes immediately
  before launch.
- Free GPU means A100, `memory.used <= 1024 MiB`, `utilization.gpu <= 10%`,
  and no foreign active compute process.
- For this autoresearch loop, set `MAX_GPUS=4` by default.
- If a package supports variable GPU count, use up to four free GPUs and label
  the result with the actual `NxA100`.
- If a package hard-requires an exact GPU count greater than four, stop and ask
  Krish before changing the protocol.
- Run detached with `tmux`, Slurm, `nohup`, or `setsid`; never leave a long run
  only in an interactive SSH terminal.
- Always record `HOST_GPU_IDS`, `GPU_COUNT`, dense container CUDA IDs, Docker
  image, commit SHA, config path, command, and claim boundary.

## Iteration Artifacts

Each loop iteration should produce or update:

```text
outputs/autoresearch/<UTC-iteration-id>/
  iteration.md
  sources.jsonl
  feature_spec.md
  test_log.txt
  claim_gate.json
  cluster_protocol.log        # only when A100 was used
  evidence_bundle.zip         # only when packaging is required
```

Committed docs should summarize the evidence and link to small artifacts. Large
checkpoints, raw arrays, raw public data, cluster caches, secrets, `.env*`,
`*.pem`, `*.key`, tarballs, and nested zips stay out of git.

Use `docs/templates/autoresearch_iteration.md` as the report skeleton.

## Completion Checklist

Before an iteration is complete, verify:

- Current branch and commit SHA are recorded.
- Required first-read docs were checked.
- Feature has a claim boundary and failure criteria.
- `claim_scope` and `stop_reason` are present in benchmark-facing artifacts.
- GatePass fields are passed or honestly marked failed.
- Aggregate metrics are separated from task/sidecar metrics.
- Exercise/depression gate was applied if relevant.
- Local tests or dry runs passed, or failures are preserved with exact logs.
- A100 runs used actual free GPUs and truthful `NxA100` labels.
- Evidence bundle contains no secrets, checkpoints, raw arrays, tarballs, or
  nested zips.
- Final report says what changed, what evidence supports it, what remains
  unproven, and the next candidate.
