# Kahlus-STF Public Dataset Review

This is a source-selection note for the first public-data STF smoke. It is not a
clinical validation claim and does not permit A100 training by itself.

## First Target: CHB-MIT Scalp EEG Database

Source: <https://physionet.org/content/chbmit/1.0.0/>

Reason for first use:

- public scalp EEG epilepsy dataset on PhysioNet
- record-level lists are available through `RECORDS` and `RECORDS-WITH-SEIZURES`
- seizure annotation sidecars use `.edf.seizures`
- common benchmark target in seizure detection/prediction papers

Local audit command:

```bash
PYTHONPATH=src python3 scripts/fetch_chb_mit_smoke_subset.py \
  --dataset chb_mit_physionet \
  --out-root /tmp/kahlus_chbmit_smoke_subset \
  --patients 2 \
  --records-per-patient 2
```

The fetch command materializes a small CHB-MIT subset outside the repository. It
downloads only selected EDFs, `RECORDS`, `RECORDS-WITH-SEIZURES`, patient summary
text files, and required `.edf.seizures` sidecars.

```bash
PYTHONPATH=src python3 scripts/run_stf_public_data_audit.py \
  --dataset chb_mit_physionet \
  --data-root /tmp/kahlus_chbmit_smoke_subset \
  --out-dir /tmp/kahlus_stf_chbmit_audit
```

The audit checks metadata shape only. It does not parse EDF, download data, copy
raw signals, or launch A100 jobs.

Local EDF smoke command after the audit passes:

```bash
PYTHONPATH=src python3 scripts/run_stf_chb_mit_smoke.py \
  --dataset chb_mit_physionet \
  --data-root /tmp/kahlus_chbmit_smoke_subset \
  --out-dir /tmp/kahlus_stf_chbmit_smoke \
  --max-records 4 \
  --max-samples-per-record 900000 \
  --max-channels 8
```

The smoke reads capped local EDF records with `edfio`, preserves variable-length
record windows, builds patient-held-out plus time-held-out tasks, and reports
persistence, ridge-AR, TinySSM, shuffled-target control, cycle/time-of-day,
event-frequency, logistic-ridge, and time-shifted-label rows when CHB-MIT
summary text files are present. Binary `.edf.seizures` parsing remains out of
scope; summary text is the event-interval source for this smoke.

## Literature Stop

The research risk is not lack of papers; it is over-optimistic validation.
Shafiezadeh et al. report that patient-independent validation is rare in EEG
seizure prediction, and Wong et al. emphasize that public EEG datasets differ in
structure enough to hurt reproducibility. Therefore, Kahlus-STF starts with
patient-held-out and time-held-out audits before any neural architecture upgrade.

## Blocked

- committing raw EDF or annotation files
- diagnosis, treatment, medication, stimulation, seizure-prevention, or vEEG/PSG
  replacement claims
- A100 runs before local synthetic and public-data smokes pass
