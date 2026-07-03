# Kahlus-NeuroVisual Roadmap

## Need Statement

A way is needed to structure and quantify brief visual-perceptual episodes in people with prior seizure history without claiming diagnosis.

This branch is future work. It must not support self-triggering, unsupervised photic testing, diagnosis, or treatment.

## Safe Initial Scope

- retrospective public EEG data review
- event-schema design for visual-perceptual episode annotation
- non-diagnostic signal quality and episode-structure metrics
- clinician/researcher-supervised workflow assumptions for any future real protocol

## Dataset Candidates To Investigate

- CHB-MIT
- TUH EEG / TUSZ
- Siena EEG
- MOABB

## Baselines

- seizure/event labels as metadata only when dataset terms allow
- persistence and spectral baselines
- ridge/TCN/Transformer EEG baselines
- identity and leakage probes

## Evaluation Criteria

- held-out subject/site split where supported
- no photic stimulation instructions
- no diagnosis/treatment wording
- finite metrics
- explicit quality flags
- evidence gate blocks epilepsy diagnosis claims

## Buildable In 8 Weeks

- dataset/license review table
- safe episode schema draft
- synthetic annotation fixture
- no hardware, no app, no triggering task

## Long-Term

- supervised research annotation tool
- linkage to v1/v2 representations after strict evidence review
- prospective protocol only after appropriate review and oversight

## Citations To Verify

| Reference | Why It Matters | citation_status |
| --- | --- | --- |
| CHB-MIT | public seizure EEG candidate | needs_verification |
| TUH/TUSZ | public clinical EEG candidate | needs_verification |
| Siena EEG | seizure EEG candidate | needs_verification |
| MOABB | non-clinical EEG benchmark tooling | needs_verification |
