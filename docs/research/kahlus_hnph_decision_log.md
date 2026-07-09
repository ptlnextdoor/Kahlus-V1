# Kahlus HNPH Decision Log

**Protocol:** `kahlus_hnph_full_development_protocol.md`
**Rule:** Any decision that changes a target, split, metric, baseline, calibration method, dataset, model family, external-test seal, or claim boundary must be entered here before code or results relying on it are merged.

## Entry Template

```markdown
## D-YYYY-MM-DD-NN: <short decision title>

**Status:** proposed | accepted | superseded | rejected
**Owner:**
**Protocol version:**
**External test opened:** no | yes

### Decision

<What is being frozen, changed, or rejected?>

### Evidence Available at the Time

<Links to tests, audits, data cards, paper evidence, or issue/PR discussion.>

### Alternatives Considered

| Alternative | Decision | Reason |
| --- | --- | --- |
| | | |

### Consequences

- Required code or documentation changes:
- Required reruns:
- Claim impact:
- External-seal impact:

### Approval Record

| Role | Name | Date | Decision |
| --- | --- | --- |
| Scientific owner | | | |
| Statistics/review owner | | | |
| Implementation owner | | | |
```

## Initial Entries

### D-2026-07-09-01: Adopt HNPH/KSTF as the Phase 0 Program

**Status:** accepted
**Owner:** Kahlus project owner
**Protocol version:** 0.1.0
**External test opened:** no

### Decision

Reframe Phase 0 around the Human Neural Predictability Horizon and the Sleep Transition Frontier. The first question is whether a causal EEG history adds calibrated, externally generalizable information about state-scale dynamics and the next stable Wake/NREM/REM transition beyond named linear and semi-Markov baselines.

### Evidence Available at the Time

- `docs/research/kahlus_hnph_full_development_protocol.md`
- `docs/research/kahlus_hnph_pr_delivery_plan.md`

### Consequences

- The Neural Field Compiler is a model candidate, not the program claim.
- The first result must be a baseline-only evidence packet.
- No clinical, seizure-warning, diagnostic, treatment, digital-twin, or foundation-model claim is allowed.
- Existing Kahlus infrastructure is reused only where its contract survives the new evaluator.

### D-2026-07-09-02: Freeze the Initial Delivery Shape

**Status:** accepted
**Owner:** Kahlus project owner
**Protocol version:** 0.1.0
**External test opened:** no

### Decision

Use short-lived `main`-based PRs, allowing a temporary two-PR stack only for a direct unmerged contract dependency.

### Consequences

- Existing PR #46 is reviewed and merged as an independent prerequisite.
- The DDP lockstep fix is rebased and reopened against `main` rather than merged through the old stack.
- Oversized failing PRs #44 and #45 are superseded and may be selectively salvaged later.
