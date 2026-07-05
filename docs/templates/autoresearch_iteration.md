# Autoresearch Iteration Report

## Header

- Iteration ID:
- Date:
- Branch:
- Commit SHA:
- Conductor:
- Feature:
- Claim boundary:
- Claim scope:
- Stop reason:
- Actual GPU label:

## Required First Reads

- [ ] `AGENTS.md`
- [ ] `README.md`
- [ ] `docs/CLAIMS.md`
- [ ] `docs/ROADMAP.md`
- [ ] `docs/research/neurotwin_project_state.md`
- [ ] `docs/research/neurotwin-technical-report.md`
- [ ] `/Users/krishgarg/Documents/papers/neuroML/cluster/A100_CLUSTER_AGENT_RUNBOOK.md`

## Research Evidence

| Source | ID | Design | Finding | Limitation | Claim impact |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

Exercise/depression gate applied: `yes/no/not relevant`

If applied, record the allowed wording and safety boundary:

```text

```

## Feature Spec

- Roadmap gate advanced:
- User-visible behavior:
- Files changed:
- Tests planned:
- Failure criteria:

## Implementation Notes

```text

```

## Verification

| Check | Command or artifact | Result |
| --- | --- | --- |
| Local tests |  |  |
| Dry run |  |  |
| A100 preflight |  |  |
| Evidence bundle scan |  |  |

## Kahlus Gate Predicate

```text
GatePass = C_split ∧ C_finite ∧ C_baseline ∧ C_controls ∧ C_power ∧ C_claim_scope
```

| Gate field | Evidence | Result |
| --- | --- | --- |
| `C_split` | held-out subject/site/dataset/modality audit |  |
| `C_finite` | finite payload and metrics check |  |
| `C_baseline` | best-baseline comparison, including moving average/persistence where relevant |  |
| `C_controls` | shuffle, time-shift, permutation null, subject probe, synthetic NULL |  |
| `C_power` | event count, positive windows, clusters, bootstrap count |  |
| `C_claim_scope` | `claim_scope` and `stop_reason` in artifacts |  |

Aggregate-vs-sidecar metric separation:

```text

```

## Cluster Protocol

Only fill this section when A100 was used.

- SSH/runbook used:
- `HOST_GPU_IDS`:
- `GPU_COUNT`:
- Dense container CUDA IDs:
- Docker image:
- Config:
- Launch command:
- Log path:

## Claim Gate

```json
{
  "claim_scope": "",
  "stop_reason": "",
  "gate_pass": false,
  "c_split": false,
  "c_finite": false,
  "c_baseline": false,
  "c_controls": false,
  "c_power": false,
  "c_claim_scope": false,
  "scientific_claim_allowed": false,
  "model_superiority_claim_allowed": false,
  "clinical_claim_allowed": false,
  "reason": ""
}
```

## Outcome

- Status:
- Evidence summary:
- Remaining uncertainty:
- Next best candidate:
