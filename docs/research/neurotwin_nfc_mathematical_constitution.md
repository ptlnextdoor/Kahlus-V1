# NeuroTwin NFC Mathematical Constitution

This is the canonical markdown entry point for the NeuroTwin NFC math. The longer LaTeX/PDF dossier remains in `docs/research/neurotwin_nfc_research_dossier.tex` and `docs/research/neurotwin_nfc_research_dossier.pdf`.

## Primitive

NFC treats each recording as a partial, noisy observation of a subject-specific latent neural field:

```{math}
F_s(x,t,\omega) \in \mathbb{R}^d
```

Here `s` indexes subject, `x` indexes neural location or parcel, `t` indexes time, and `omega` captures stochastic state.

## Observation Operator

Each modality is an observation operator over the same field:

```{math}
Y_m = \mathcal{O}_m(F_s,A_s,U,\epsilon_m)
```

`A_s` is anatomy or subject geometry, `U` is stimulus/task/context, and `epsilon_m` is modality-specific noise.

## Generative Model

```{math}
p(Y_{1:M}\mid U,A_s)=\int p(F\mid U,A_s)\prod_m p(Y_m\mid F,A_s)\,dF
```

The benchmark question is whether a field-mediated model explains held-out observations better than direct translation baselines.

## Controlled Dynamics

```{math}
F(t+\Delta)=\Phi_{\theta,\Delta}(F(t),U_{[t-H,t]},A_s)
```

This says future field state depends on current field state, recent stimulus/context history, and subject anatomy.

## Neural Field Dynamics

```{math}
\tau \partial_t F(x,t)
=
-F(x,t)
+
\int_\Omega K_\theta(x,x',t)\sigma(F(x',t))\,dx'
+
B_\theta U(t)
+
\eta(x,t)
```

## Discretized Dynamics

```{math}
Z_{t+1}=Z_t+\Delta t[-DZ_t+K_t\sigma(Z_t)+BU_t]+\xi_t
```

## Low-Rank Pair Kernel

```{math}
K_t \approx U_tV_t^\top
```

```{math}
M_t=\operatorname{softmax}\left((U_tV_t^\top)/\sqrt r+S\right)
```

```{math}
Z'_t=Z_t+M_tZ_tW
```

This is where the old Pair-Operator idea survives: not as the main architecture, but as a low-rank relational field-update ablation.

## fMRI Observation

```{math}
(H_{\mathrm{HRF}}a)(t)=\int_0^\infty h(\tau)a(t-\tau)\,d\tau
```

```{math}
Y_{\mathrm{fMRI}}(p,t)=R_pH_{\mathrm{HRF}}g_\theta(F(\cdot,t))+\epsilon
```

## EEG and MEG Observation

```{math}
Y_{\mathrm{EEG}}(t)=L_sJ_\theta(F_t)+\epsilon_{\mathrm{EEG}}
```

```{math}
Y_{\mathrm{MEG}}(t)=M_sJ_\theta(F_t)+\epsilon_{\mathrm{MEG}}
```

## Spike, Calcium, and Behavior Observations

```{math}
Y_{n,t}\sim\operatorname{Poisson}\left(\Delta t\cdot\operatorname{softplus}(w_n^\top F(x_n,t))\right)
```

```{math}
c_{n,t}=(k_{\mathrm{Ca}}*r_n)(t)+\epsilon
```

```{math}
p(a_t\mid F_t,U_t)=\operatorname{softmax}(C\operatorname{pool}(F_t)+DU_t)
```

These are theory entries unless corresponding adapters and tests are explicitly implemented.

## Operator Learning Interpretation

NFC is not direct modality fusion. It learns an inverse path from observations to latent field state and a forward path from field state to modality-specific readouts.

## State-Space Interpretation

NFC can be read as a controlled state-space model where `F_t` is the latent state and each modality supplies a partial observation channel.

## Graph Calculus Interpretation

```{math}
\nabla_wF(i,j)=\sqrt{w_{ij}}(F_j-F_i)
```

```{math}
L=D-W
```

```{math}
R_{\mathrm{graph}}(F)=\sum_{(i,j)\in E}w_{ij}\|F_i-F_j\|^2=\operatorname{Tr}(F^\top L F)
```

## Identifiability and Gauge Ambiguity

```{math}
F'=AF
```

```{math}
\mathcal{O}'_m=\mathcal{O}_mA^{-1}
```

```{math}
\mathcal{O}'_m(F')=\mathcal{O}_m(F)
```

Latent fields are identifiable only up to transformations unless constrained by architecture, observations, and regularizers.

## Uncertainty and Calibration

```{math}
\mathcal{L}_{\mathrm{NLL}}
=
\sum_{m,t,i}
\frac{(y_{m,t,i}-\mu_{m,t,i})^2}{2\sigma_{m,t,i}^2}
+
\frac12\log\sigma_{m,t,i}^2
```

```{math}
\mathbb{P}[Y\in C_\alpha(X)]\approx 1-\alpha
```

Uncertainty artifacts are not claim evidence unless they use actual uncertainty outputs and a documented calibration target.

## Synthetic Proving Ground

The NFC synthetic suite is a gate, not a result. It must test true field-grounded tasks, no-observation and no-pair ablations, no NaNs, strict shape contracts, and no target leakage.

## Why Direct Fusion Is Not Enough

Direct fusion can predict one modality from another, but it does not force a shared field explanation. NFC's scientific bet is that field-mediated translation is more robust under held-out subject/site/dataset splits.

## Why Pair-Operator Is a Submodule

Pair-Operator captures low-rank relational updates. NFC needs that idea only as one possible kernel inside a broader latent-field and observation-operator model.
