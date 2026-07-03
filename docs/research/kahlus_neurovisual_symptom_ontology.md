# Kahlus NV-1 Neurovisual Symptom Ontology

NV-1 is a Kahlus side branch for structured neurovisual symptom mapping. It does not replace Kahlus v1 EEG forecasting, Kahlus v2 multimodal observation-operator modeling, Kahlus v3 perturbation-response modeling, or ResearchDock.

This ontology is not a diagnosis, not medical advice, not seizure prediction, not epilepsy detection, not migraine aura detection, and not a replacement for clinician review.

## Purpose

The NV-1 ontology converts a generic neurovisual episode history into structured fields that can later map into Kahlus v2/v3 as `structured_history_h_t`. The output is for clinician or researcher review alongside proper medical evaluation.

## Ontology Fields

Timing/course: `onset_speed`, `duration_seconds`, `episode_frequency`, `course_change_recent`.

Awareness/memory: `awareness_retained`, `memory_retained`, `impaired_awareness_flag`.

Visual features: `visual_field_location`, `color_distortion`, `shape_or_object_distortion`, `pattern_or_outline_effect`, `motion_or_flicker`, `expansion_or_spreading`, `light_or_glare_sensitivity`, `screen_or_sun_trigger`, `moving_object_tracking_trigger`.

Perceptual/self symptoms: `derealization`, `depersonalization`, `body_detachment`, `alarm_or_impending_doom`, `neck_or_head_sensation`.

Associated symptoms: `headache`, `photophobia`, `nausea`, `confusion_after`, `fatigue_after`, `motor_symptoms`, `speech_symptoms`.

Context/confounders: `prior_seizure_history`, `migraine_history`, `concussion_history`, `medication_context`, `caffeine_or_stimulant_context`, `sleep_context`, `hydration_context`, `stress_context`.

Negatives: `no_new_objects_seen`, `no_minutes_long_progression`, `no_loss_of_consciousness`, `no_postictal_confusion_reported`, `no_motor_event_reported`.

Safety and review fields: `urgent_red_flags`, `clinician_questions`, `should_seek_medical_evaluation`, `not_diagnosis_notice`.

## Synthetic Example Profile

```json
{
  "onset_speed": "abrupt",
  "duration_seconds": 20,
  "awareness_retained": true,
  "memory_retained": true,
  "visual_field_location": "peripheral",
  "motion_or_flicker": true,
  "screen_or_sun_trigger": true,
  "no_new_objects_seen": true,
  "no_loss_of_consciousness": true
}
```

This example is synthetic and generic. It contains no personal medical details, dates, locations, clinician names, or private records.

## Missing Clinician Questions

The intake builder emits missing clinician questions for absent duration, awareness, memory, headache, photophobia, nausea, medication/substance context, and sleep context fields.

## Red-Flag Checklist

The intake builder records clinician-facing red flags such as prior seizure history, impaired awareness, motor symptoms, speech symptoms, and recent course change. These flags are not patient-facing self-administration instructions and do not assert safety.

## Condition Matrix Summary

The condition comparison matrix includes occipital focal aware seizure visual aura, reflex/photosensitive seizure activity, migraine aura, visual snow syndrome, concussion-related visual symptoms, panic/derealization episodes, functional/psychogenic episodes, and medication/substance/metabolic contributors.

The matrix is research framing only. It is not a diagnosis engine.

## Kahlus v2/v3 Mapping

NV-1 episode phenotype profiles are intended to become structured history inputs for future observation-operator or perturbation-response work. The current sprint does not train a model, implement event-window forecasting, run A100, or make clinical claims.

## Deferred Model Experiment

A future sprint may evaluate retrospective event-aligned EEG window forecasting around annotated seizure events in CHB-MIT or TUSZ after adapter review. That future task is research-only and must not become prospective seizure prediction, epilepsy detection, patient alerting, diagnosis, clinical triage, or treatment guidance.
