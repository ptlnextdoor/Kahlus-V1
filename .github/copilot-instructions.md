## NeuroTwin Repo Instructions

Before answering architecture or codebase questions, read `graphify-out/GRAPH_REPORT.md` if it exists.
If `graphify-out/wiki/index.md` exists, navigate it for deep questions.
Type `/graphify` in Copilot Chat to build or update the knowledge graph.

Project stance:

- NeuroTwin v1 is a leakage-proof Neural Translation benchmark and scaffold.
- Do not claim first brain foundation model, first multimodal brain model, first stimulus-to-brain model, or clinical digital twin.
- Treat TRIBE v2, BrainVista, Brain-OF, BrainOmni, and Brain Harmony as explicit competitors/baselines.
- Keep raw neural data out of the repo; commit manifests, adapters, configs, reports, and license notes only.
- After code changes, run `PYTHONPATH=src python3 -m unittest discover -s tests -v`.
- After code or architecture changes, run `graphify update .`.
