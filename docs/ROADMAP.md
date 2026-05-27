# Roadmap

1. Local CPU smoke and prepared-manifest plumbing are locked.
2. Locked MOABB smoke path is available; scientific use still requires running it with optional dependencies and data access.
3. BIDS/OpenNeuro support is derivative-only with validation; raw fMRI preprocessing remains out of scope for v1.
4. Prepared-manifest training supports single-task and `neural_translation_v1` multi-task runs.
5. A100 is the canonical cluster target; H100 remains a compatible high-memory variant.
6. Next acceptance gate: three real MOABB reports with held-out subjects, val-selected checkpoints, final test metrics, and baseline failures/statuses recorded.
7. Draft paper only after real benchmark results exist.
