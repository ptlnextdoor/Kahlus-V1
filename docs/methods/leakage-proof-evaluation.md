# Leakage-Proof Evaluation

NeuroTwin v1 treats split construction as part of the scientific claim.

Rules:

- Build `SplitManifest` from recording-level metadata before preprocessing, augmentation, or windowing.
- Use held-out subject, site, and dataset splits as primary tests.
- Run record-ID reuse checks for every split.
- For subject-held-out claims, no subject can appear across train/val/test.
- For site-held-out claims, no site can appear across train/val/test.
- For dataset-held-out claims, no dataset can appear across train/val/test.
- Clinical prediction is secondary and cannot rescue a weak Neural Translation result.

Main failure modes to catch:

- Subject identity leakage.
- Scanner/site leakage.
- Repeated-session leakage.
- Stimulus or clip leakage.
- Metadata labels that encode the target.
- Training windows generated before split assignment.
