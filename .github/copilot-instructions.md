## Kahlus / NeuroTwin — agent grounding

**Read [AGENTS.md](../AGENTS.md) before any code change or scientific claim.** Conductor workspaces: [CONDUCTOR.md](../CONDUCTOR.md).

### TL;DR

- Kahlus = leakage-controlled residual forecastability engine (Neural-CASP + RFS in bits).
- Flagship: Passive PCI. Not clinical, not foundation model, not 3.116 MSE headline.
- Gates: held-out splits, beat best baseline (incl. moving average), controls, artifacts on disk.
- Raw neural data never committed.

### Graphify

Before architecture questions, read `graphify-out/GRAPH_REPORT.md`. After code changes: `graphify update .`.

### NeuroTwin v1 (narrower legacy scope)

- Leakage-proof Neural Translation under held-out subject/site/dataset splits.
- Do not claim first brain foundation model, first multimodal model, or clinical digital twin.
- Brain-OF is the primary multimodal opponent; TRIBE v2 and BrainVista are baselines.
