# HNPH Estimand And Information-Theoretic Contract

## Status

This note freezes the mathematical interpretation of Phase 0. The decomposition below is an application of standard conditional information and proper-scoring identities. It is not presented as a new theorem. Any future novelty claim requires a separate literature review and peer scrutiny.

## Scientific Object

For subject `s` and an eligible issue time `a`, define:

- `H`: oracle sleep-history covariates available at issue time, including current scored state, two preceding states, bout age, elapsed night time, and recent transitions;
- `X`: causal EEG history ending at the issue time;
- `Y`: the future competing-risk observation, consisting of destination state and transition-time band, or a right-censoring outcome under the frozen ontology;
- `q0(Y | H)`: the chief semi-Markov comparator;
- `q1(Y | X, H)`: the EEG-augmented forecaster.

The subject-balanced log-skill estimand in bits is

```text
Delta(q1, q0)
  = E_subject [ E_anchor|subject [ log2 q1(Y | X,H) - log2 q0(Y | H) ] ].
```

Subjects, not windows, receive equal outer weight. Event and censoring probabilities must be computed from one normalized competing-risk distribution over the same observable outcome space.

## Proposition 1: Predictive-Gain Decomposition

Let `P` be the true joint distribution, and assume `q0` and `q1` assign nonzero probability wherever `P` does. Then

```text
Delta(q1, q0)
  = I_P(Y; X | H)
    - E_X,H[ KL(P(Y | X,H) || q1(Y | X,H)) ] / ln(2)
    + E_H[ KL(P(Y | H) || q0(Y | H)) ] / ln(2).
```

### Proof

Add and subtract the true conditional log probabilities inside the expected log ratio:

```text
E ln(q1/q0)
  = E ln(P(Y|X,H)/P(Y|H))
    + E ln(q1/P(Y|X,H))
    + E ln(P(Y|H)/q0).
```

The first term is `I_P(Y;X|H)`. Conditioning on `(X,H)`, the second is the negative conditional Kullback-Leibler divergence from the true EEG-conditional law to `q1`. Conditioning on `H`, the third is the divergence from the true history-only law to `q0`. Division by `ln(2)` converts nats to bits. This proves the identity.

## Consequences

1. If `q0=P(Y|H)` and `q1=P(Y|X,H)`, then `Delta=I_P(Y;X|H)`.
2. If EEG adds no conditional information and `q0` is oracle, then every `q1` has `Delta<=0`.
3. A positive empirical `Delta` does not by itself prove EEG information. It can be inflated by the comparator-misspecification term.
4. A more expressive neural model can score worse despite real EEG information if its conditional estimation error exceeds the available information.

This is why the semi-Markov comparator, nuisance controls, matched budgets, and external cohort are part of the estimand rather than optional baselines.

## Competing-Risk Likelihood

For time band `j`, destination `k`, total event hazard `h_j=sum_k h_jk`, and survival through the preceding band

```text
S_0 = 1,
S_j = product_{u=1..j} (1 - h_u).
```

An observed event in `(j,k)` contributes

```text
-log(S_{j-1} h_jk),
```

while right censoring after band `c` contributes

```text
-log(S_c).
```

Probability flooring is a numerical device only: probabilities are floored at the frozen epsilon and renormalized before scoring. Any implementation that changes censor indexing, floors without renormalizing, or compares models on different observable outcomes violates the estimand.

## Horizon Decision Functional

For each preregistered horizon band `b`, let `L_b` be the one-sided simultaneous 95% lower confidence bound for subject-level log skill. Let every negative control have simultaneous upper bound `U_b^control`. A band is supported only when:

```text
L_b > 0,
max_control U_b^control <= 0,
external Brier is no worse than the chief comparator,
and calibration plus prediction-set gates pass.
```

The predictability frontier is the latest supported band. If no band is supported, the frontier is explicitly empty. The procedure must not interpolate a positive horizon between failed bands.

## Identifiability Boundary

Phase 0 identifies predictive value relative to measured history and a declared comparator class. It does not identify a cortical source, causal mechanism, seizure risk, diagnosis, treatment effect, or deployable warning system. CAP is an external cohort, but independent replication requires a third cohort because protocol design and analysis decisions are informed by Sleep-EDF and CAP.

## Architecture Implication

The scientific target is `Delta`, not latent-state complexity. The smallest model that estimates `q1` reliably is preferred. Architecture expansion stops if the semi-Markov comparator ties or wins, if negative controls remain positive, or if external calibration fails. A Neural Field Compiler becomes relevant only after the conditional-information signal exists and coordinate/montage experiments can falsify a field-specific advantage.

## Related Evidence Boundary

- [SSF-SET](https://pubmed.ncbi.nlm.nih.gov/41662557/) establishes that future sleep-stage forecasting is not empty territory.
- [U-Sleep](https://www.nature.com/articles/s41746-021-00440-5) motivates heterogeneous external sleep evaluation but does not answer the frozen conditional-information question.
- The seizure-forecasting literature reports useful signals alongside substantial validation and standardization gaps; it motivates strict external validation, not a seizure claim for HNPH.
