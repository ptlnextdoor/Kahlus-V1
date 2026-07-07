# TurboQuant/TurboVec Retrieval Notes

TurboQuant/TurboVec is optional retrieval, compression, and audit infrastructure around NFC. It is not the core NeuroTwin model contribution and must not become a required dependency.

## Math Summary

Normalize:

```{math}
r=\|v\|_2,\qquad u=\frac{v}{\|v\|_2}
```

Random rotation:

```{math}
z=Ru
```

High-dimensional coordinate approximation:

```{math}
z_i\approx\mathcal{N}(0,1/d)
```

Quantization map:

```{math}
Q:\mathbb{R}^d\rightarrow\{0,1\}^{bd}
```

MSE objective:

```{math}
\mathbb{E}\|x-\hat{x}\|_2^2
```

Inner-product distortion objective:

```{math}
\mathbb{E}\left[(\langle x,q\rangle-\langle\hat{x},q\rangle)^2\right]
```

Score error bound:

```{math}
|\langle q,z_i\rangle-\langle q,\hat z_i\rangle|
\leq
\|q\|_2\|z_i-\hat z_i\|_2
```

## Retrieval-kNN Baseline

```{math}
\mathcal{N}_k(q)=\{i_1,\ldots,i_k\}
```

```{math}
\hat y_{\mathrm{test}}=
\sum_{i\in \mathcal{N}_k(q)}w_i y_i
```

```{math}
w_i=
\frac{\exp(\tau\langle q,z_i\rangle)}
{\sum_{j\in\mathcal{N}_k(q)}\exp(\tau\langle q,z_j\rangle)}
```

This baseline may use train labels and test query features. It must never use test targets.

## Semantic Near-Duplicate Audit

```{math}
d_{\min}(x_{\mathrm{test}},\mathcal{D}_{\mathrm{train}})
=
\min_{x_i\in\mathcal{D}_{\mathrm{train}}}
\|q(x_{\mathrm{test}})-q(x_i)\|
```

This can flag train/test semantic leakage in stimulus features, especially for Algonauts/CNeuroMod.

## Why It Could Help

- Compress large stimulus-feature stores.
- Support exact or approximate nearest-neighbor retrieval baselines.
- Audit semantic near-duplicates between train and test stimuli.
- Store latent field summaries for memory/debug workflows.

## Risks

- Low-dimensional vectors can violate high-dimensional approximations.
- Approximate retrieval can miss duplicates.
- Quantization distortion can change rankings.
- Optional dependency friction can break reproducibility.
- Quantization is not automatically differentiable or claim-relevant.

## Implementation Priority

1. Numpy exact vector store.
2. Optional lazy TurboVec adapter.
3. Retrieval baseline.
4. Semantic duplicate audit.

No implementation is added in this pass.
