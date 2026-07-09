## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `graphify update .` to keep the graph current (AST-only, no API cost)

## NeuroTwin claim hygiene

- The core v1 claim is leakage-proof Neural Translation: missing-modality reconstruction, future-state forecasting, and few-shot subject adaptation under held-out subject/site/dataset splits.
- Do not claim first brain foundation model, first multimodal brain model, first stimulus-to-brain model, or clinical digital twin.
- Brain-OF is the primary multimodal boss fight; TRIBE v2 and BrainVista are stimulus-to-fMRI baselines, not empty territory.
- Raw public neural data must not be committed.

## Kaggle / Composio access for dataset discovery

- Kaggle is available through the Composio CLI on this machine. Do not block on the Rube-backed `kaggle-automation` MCP skill if those tools are not exposed in a Codex session.
- Before Kaggle discovery commands, ensure:

```bash
export PATH="$HOME/.composio:$PATH"
```

- Verified read-only Kaggle search path:

```bash
composio execute KAGGLE_LIST_DATASETS -d '{"search":"sleep edf","page":1}'
composio execute KAGGLE_LIST_DATASETS -d '{"search":"CHB MIT EEG seizure","page":1}'
composio execute KAGGLE_LIST_DATASETS -d '{"search":"sleep EEG PSG","page":1}'
```

- Useful discovery commands:

```bash
composio tools list kaggle --limit 80
composio search "kaggle eeg sleep seizure" --human --limit 20
```

- Treat Kaggle datasets as convenience mirrors unless the owner/source is official. Official dataset sources such as PhysioNet, OpenNeuro, NSRR, ISRUC, MASS, TUH/TUSZ, and original papers remain the evidence source of truth.
