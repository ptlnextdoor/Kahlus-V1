from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.forecastability.m1 import TransitionFixture, _run_fixture_with_traces, make_transition_fixture
from neurotwin.forecastability.m2 import (
    SLEEP_EDF_BASE_URL,
    _as_transition_fixture,
    _local_sleep_edf_pairs,
    _sha256,
    load_sleep_edf_fixture,
)


DEFAULT_HORIZONS = (1, 2, 3)
NUISANCE_PROBE_KEYS = ("patient", "site", "time_bucket", "session")
NUISANCE_PROBE_MARGIN = 0.20
CLUSTER_PERMUTATION_ALPHA = 0.05
SLEEP_EDF_MIN_PRIMARY_POSITIVE_EVENTS = 100
M4_SLEEP_EDF_PREREGISTRATION_SCHEMA = "kahlus.forecastability.m4.sleep_edf_preregistration.v1"
M4_SLEEP_EDF_EXECUTION_SCHEMA = "kahlus.forecastability.m4.sleep_edf_primary_execution.v1"


def run_m4_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    sleep_edf_root: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    primary = _resolve_primary_horizon(horizons, primary_horizon)
    known = _curve_payload(make_transition_fixture(seed=seed, residual_signal=True), horizons=horizons, seed=seed)
    null = _curve_payload(make_transition_fixture(seed=seed + 100, residual_signal=False), horizons=horizons, seed=seed + 100)
    sleep = _sleep_edf_curve_payload(sleep_edf_root, horizons=horizons, seed=seed + 200)
    synthetic_passed = _known_curve_passes(known, primary_horizon=primary) and _null_curve_passes(
        null,
        primary_horizon=primary,
    )
    sleep_failures = _sleep_curve_failures(sleep, primary_horizon=primary)
    gate = {
        "milestone": "M4",
        "method": "leakage_safe_horizon_wise_label_curve",
        "horizons": list(horizons),
        "primary_horizon": primary,
        "synthetic_known_signal": known,
        "synthetic_null": null,
        "sleep_edf_smoke": sleep,
        "sleep_edf_smoke_failures": sleep_failures,
        "synthetic_gate_passed": synthetic_passed,
        "gate_passed": bool(synthetic_passed and not sleep_failures),
        "claim_scope": "benchmark_method_only_not_clinical_or_foundation_model_claim",
        "stop_reason": "M4 gate reached; full claim requires powered public-data and external-dataset validation.",
    }
    _write_json(out / "m4_gate_report.json", gate)
    _write_report(out / "M4_EVIDENCE_REPORT.md", gate)
    return gate


def build_m4_sleep_edf_preregistration(
    *,
    seed: int = 5,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    primary_horizon: int | None = None,
    sleep_edf_max_pairs: int | None = None,
) -> dict[str, Any]:
    planned_horizons = tuple(int(horizon) for horizon in horizons)
    primary = _resolve_primary_horizon(planned_horizons, primary_horizon)
    if sleep_edf_max_pairs is not None and int(sleep_edf_max_pairs) <= 0:
        raise ValueError("sleep_edf_max_pairs must be positive or None")
    return {
        "schema": M4_SLEEP_EDF_PREREGISTRATION_SCHEMA,
        "protocol": "M4 Sleep-EDF primary-horizon pre-analysis plan",
        "dataset_id": "sleep_edf_expanded",
        "dataset_url": SLEEP_EDF_BASE_URL,
        "raw_data_policy": "outside_repo_only",
        "seed": int(seed),
        "horizons": list(planned_horizons),
        "primary_horizon": int(primary),
        "sleep_edf_max_pairs": None if sleep_edf_max_pairs is None else int(sleep_edf_max_pairs),
        "primary_endpoint": "primary_horizon_rfs_bits_vs_strongest_gated_baseline",
        "inferential_scope": "primary_horizon_only",
        "descriptive_horizons": [int(horizon) for horizon in planned_horizons if horizon != primary],
        "minimum_event_patients": 8,
        "minimum_primary_positive_events": SLEEP_EDF_MIN_PRIMARY_POSITIVE_EVENTS,
        "controls": {
            "nuisance_probe_rule": f"fail if accuracy exceeds chance + {NUISANCE_PROBE_MARGIN:.2f}",
            "primary_cluster_permutation_alpha": CLUSTER_PERMUTATION_ALPHA,
            "shuffled_target_control": "must stay below half of primary RFS",
            "time_shift_control": "must stay below half of primary RFS",
        },
        "claim_boundary": "benchmark-method hardening only; no clinical, treatment, or model-superiority claim",
        "clinical_claim_allowed": False,
        "model_superiority_claim_allowed": False,
        "public_data_used": False,
        "a100_jobs_launched": False,
        "preanalysis_plan": True,
    }


def m4_preregistration_hash(preregistration: dict[str, Any]) -> str:
    encoded = json.dumps(
        preregistration,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_m4_sleep_edf_primary_execution(
    out_dir: str | Path,
    *,
    sleep_edf_root: str | Path,
    repo_root: str | Path | None = None,
    preregistration: dict[str, Any] | None = None,
    fixture_loader: Any = load_sleep_edf_fixture,
) -> dict[str, Any]:
    prereg = preregistration or build_m4_sleep_edf_preregistration()
    _validate_m4_sleep_edf_preregistration(prereg)
    horizons = tuple(int(horizon) for horizon in prereg["horizons"])
    primary = _resolve_primary_horizon(horizons, int(prereg["primary_horizon"]))
    root = Path(sleep_edf_root).expanduser().resolve()
    repo = _default_repo_root() if repo_root is None else Path(repo_root).expanduser().resolve()
    if root == repo or root.is_relative_to(repo):
        raise ValueError("Sleep-EDF root must stay outside the repository")

    pairs = _local_sleep_edf_pairs(root)
    max_pairs = prereg.get("sleep_edf_max_pairs")
    used_pairs = pairs if max_pairs is None else pairs[: int(max_pairs)]
    fixture = _as_transition_fixture(fixture_loader(root, max_pairs=max_pairs))
    curve = _curve_payload(fixture, horizons=horizons, seed=int(prereg["seed"]) + 200)
    sleep_payload = {"status": "completed_sleep_edf_smoke", **curve}
    failures = _sleep_curve_failures(sleep_payload, primary_horizon=primary)
    fingerprints = _pair_fingerprints(used_pairs)
    execution = {
        "schema": M4_SLEEP_EDF_EXECUTION_SCHEMA,
        "dataset_id": prereg["dataset_id"],
        "dataset_url": prereg["dataset_url"],
        "raw_data_policy": prereg["raw_data_policy"],
        "public_data_used": True,
        "local_root_redacted": True,
        "raw_data_in_repo": False,
        "sleep_edf_pair_count": len(pairs),
        "sleep_edf_used_pairs": len(used_pairs),
        "file_fingerprints": fingerprints,
        "data_fingerprint_sha256": _fingerprint_hash(fingerprints),
        "preregistration_hash": m4_preregistration_hash(prereg),
        "preregistration": prereg,
        "horizons": list(horizons),
        "primary_horizon": primary,
        "primary_horizon_result": _primary_row(sleep_payload, primary),
        "sleep_edf_smoke": sleep_payload,
        "sleep_edf_smoke_failures": failures,
        "gate_passed": not failures,
        "claim_scope": "benchmark_method_only_not_clinical_or_foundation_model_claim",
        "clinical_claim_allowed": False,
        "model_superiority_claim_allowed": False,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / "m4_sleep_edf_preregistration.json", prereg)
    _write_json(out / "m4_sleep_edf_primary_execution.json", execution)
    return execution


def patient_horizon_labels(events: np.ndarray, patient: np.ndarray, *, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    y = np.asarray(events, dtype=np.int64)
    groups = np.asarray(patient)
    labels = {horizon: np.zeros(len(y), dtype=np.int64) for horizon in horizons}
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError("horizons must be positive")
            if len(idx) < horizon:
                continue
            if horizon == 1:
                labels[horizon][idx] = y[idx]
            else:
                labels[horizon][idx[: -(horizon - 1)]] = y[idx[horizon - 1 :]]
    return labels


def patient_horizon_valid_masks(
    events: np.ndarray,
    patient: np.ndarray,
    *,
    horizons: tuple[int, ...],
) -> dict[int, np.ndarray]:
    y = np.asarray(events, dtype=np.int64)
    groups = np.asarray(patient)
    if len(y) != len(groups):
        raise ValueError("events and patient must have the same length")
    masks = {horizon: np.zeros(len(y), dtype=bool) for horizon in horizons}
    for horizon in horizons:
        if horizon <= 0:
            raise ValueError("horizons must be positive")
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        for horizon in horizons:
            valid_count = len(idx) - max(0, horizon - 1)
            if valid_count <= 0:
                continue
            masks[horizon][idx[:valid_count]] = True
    return masks


def _curve_payload(fixture: TransitionFixture, *, horizons: tuple[int, ...], seed: int) -> dict[str, Any]:
    labels_by_horizon = patient_horizon_labels(fixture.y, fixture.patient, horizons=horizons)
    valid_masks_by_horizon = patient_horizon_valid_masks(fixture.y, fixture.patient, horizons=horizons)
    total_rows = int(len(fixture.y))
    rows = []
    for offset, horizon in enumerate(horizons):
        valid_mask = valid_masks_by_horizon[horizon]
        valid_windows = int(np.sum(valid_mask))
        if valid_windows == 0:
            raise ValueError(f"horizon {horizon} has no valid within-patient future labels")
        horizon_fixture = _filter_fixture(_with_labels(fixture, labels_by_horizon[horizon]), valid_mask)
        payload, traces = _run_fixture_with_traces(horizon_fixture, seed=seed + offset)
        nuisance_probes = payload.get("nuisance_probes", {})
        cluster_permutation = _cluster_permutation_rfs(
            traces["y"],
            traces["gated_baseline"],
            traces["logistic_full"],
            traces["patient"],
            seed=seed + 10_000 + offset,
        )
        rows.append(
            {
                "horizon": horizon,
                "total_rows": total_rows,
                "valid_rows": valid_windows,
                "evaluated_rows": payload["n"],
                "invalid_terminal_rows": total_rows - valid_windows,
                "horizon_label_policy": "drop_terminal_rows_without_within_patient_future_label",
                "rfs_bits": payload["logistic_full"]["rfs_bits"],
                "rfs_ci_low": payload["logistic_full"]["rfs_ci_low"],
                "rfs_ci_high": payload["logistic_full"]["rfs_ci_high"],
                "nll": payload["logistic_full"]["nll"],
                "gated_baseline_name": payload["gated_baseline_name"],
                "gated_baseline_nll": payload["gated_baseline_nll"],
                "positive_events": payload["positive_events"],
                "event_patients": payload["event_patients"],
                "shuffled_rfs_bits": payload["shuffled_target_control"]["rfs_bits"],
                "time_shift_rfs_bits": payload["time_shift_control"]["rfs_bits"],
                "nuisance_probes": nuisance_probes,
                "nuisance_probe_failures": _nuisance_probe_failures(
                    nuisance_probes,
                    prefix=f"horizon_{horizon}",
                ),
                "cluster_permutation": cluster_permutation,
                "cluster_permutation_failures": _cluster_permutation_failures(
                    cluster_permutation,
                    prefix=f"horizon_{horizon}",
                ),
            }
        )
    positive = [max(0.0, float(row["rfs_bits"])) for row in rows]
    return {
        "curve": rows,
        "auc_positive_rfs_bits": float(np.mean(positive)) if positive else 0.0,
        "max_rfs_bits": float(max((row["rfs_bits"] for row in rows), default=0.0)),
    }


def _sleep_edf_curve_payload(root: str | Path | None, *, horizons: tuple[int, ...], seed: int) -> dict[str, Any]:
    if root is None:
        return {"status": "not_run_no_local_sleep_edf_root"}
    try:
        fixture = _as_transition_fixture(load_sleep_edf_fixture(root, max_pairs=8))
        payload = _curve_payload(fixture, horizons=horizons, seed=seed)
        return {
            "status": "completed_sleep_edf_smoke",
            "local_root_redacted": True,
            "raw_data_policy": "outside_repo_only",
            **payload,
        }
    except Exception as exc:  # noqa: BLE001 - parser/runtime failures are evidence.
        return {
            "status": "sleep_edf_smoke_failed",
            "local_root_redacted": True,
            "raw_data_policy": "outside_repo_only",
            "error": str(exc),
        }


def _with_labels(fixture: TransitionFixture, y: np.ndarray) -> TransitionFixture:
    return TransitionFixture(
        windows=fixture.windows,
        nuisance=fixture.nuisance,
        y=np.asarray(y, dtype=np.int64),
        patient=fixture.patient,
        site=fixture.site,
        time_bucket=fixture.time_bucket,
        session=fixture.session,
    )


def _filter_fixture(fixture: TransitionFixture, mask: np.ndarray) -> TransitionFixture:
    keep = np.asarray(mask, dtype=bool)
    if len(keep) != len(fixture.y):
        raise ValueError("mask and fixture must have the same length")
    return TransitionFixture(
        windows=fixture.windows[keep],
        nuisance=fixture.nuisance[keep],
        y=fixture.y[keep],
        patient=fixture.patient[keep],
        site=fixture.site[keep],
        time_bucket=fixture.time_bucket[keep],
        session=fixture.session[keep],
    )


def _resolve_primary_horizon(horizons: tuple[int, ...], primary_horizon: int | None) -> int:
    if not horizons:
        raise ValueError("horizons must include at least one horizon")
    primary = horizons[0] if primary_horizon is None else primary_horizon
    if primary not in horizons:
        raise ValueError("primary_horizon must be present in horizons")
    return primary


def _validate_m4_sleep_edf_preregistration(preregistration: dict[str, Any]) -> None:
    if preregistration.get("schema") != M4_SLEEP_EDF_PREREGISTRATION_SCHEMA:
        raise ValueError("schema must be M4 Sleep-EDF preregistration v1")
    if preregistration.get("raw_data_policy") != "outside_repo_only":
        raise ValueError("raw_data_policy must be outside_repo_only")
    if preregistration.get("inferential_scope") != "primary_horizon_only":
        raise ValueError("inferential_scope must be primary_horizon_only")
    for key in (
        "clinical_claim_allowed",
        "model_superiority_claim_allowed",
        "public_data_used",
        "a100_jobs_launched",
    ):
        if preregistration.get(key) is not False:
            raise ValueError(f"{key} must be false in the pre-analysis plan")
    if preregistration.get("preanalysis_plan") is not True:
        raise ValueError("preanalysis_plan must be true")
    horizons = tuple(int(horizon) for horizon in preregistration.get("horizons", []))
    _resolve_primary_horizon(horizons, int(preregistration["primary_horizon"]))
    max_pairs = preregistration.get("sleep_edf_max_pairs")
    if max_pairs is not None and int(max_pairs) <= 0:
        raise ValueError("sleep_edf_max_pairs must be positive or None")


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _primary_row(payload: dict[str, Any], primary_horizon: int) -> dict[str, Any]:
    for row in payload.get("curve", []):
        if row.get("horizon") == primary_horizon:
            return row
    raise ValueError(f"primary horizon {primary_horizon} is missing from curve")


def _known_curve_passes(payload: dict[str, Any], *, primary_horizon: int = DEFAULT_HORIZONS[0]) -> bool:
    first = _primary_row(payload, primary_horizon)
    return bool(
        first["positive_events"] >= 40
        and first["event_patients"] >= 8
        and first["rfs_bits"] > 0.03
        and first["rfs_ci_low"] > 0.0
        and first["shuffled_rfs_bits"] < first["rfs_bits"] * 0.5
        and first["time_shift_rfs_bits"] < first["rfs_bits"] * 0.5
        and not first.get("cluster_permutation_failures", ["cluster_permutation_missing"])
        and _nuisance_probes_pass(payload)
    )


def _null_curve_passes(payload: dict[str, Any], *, primary_horizon: int = DEFAULT_HORIZONS[0]) -> bool:
    primary = _primary_row(payload, primary_horizon)
    return bool(
        all(
            abs(row["rfs_bits"]) < 0.03
            and row["rfs_ci_low"] <= 0.0 <= row["rfs_ci_high"]
            for row in payload["curve"]
        )
        and _cluster_permutation_is_valid(primary.get("cluster_permutation"))
        and not _cluster_permutation_is_significant(primary.get("cluster_permutation"))
        and _nuisance_probes_pass(payload)
    )


def _nuisance_probes_pass(payload: dict[str, Any]) -> bool:
    rows = payload.get("curve", [])
    return bool(rows) and all(not row.get("nuisance_probe_failures", ["nuisance_probe_missing"]) for row in rows)


def _cluster_permutation_rfs(
    y: np.ndarray,
    baseline: np.ndarray,
    pred: np.ndarray,
    patient: np.ndarray,
    *,
    seed: int,
    n_permutations: int = 300,
    alpha: float = CLUSTER_PERMUTATION_ALPHA,
) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.float64)
    baseline_proba = np.clip(np.asarray(baseline, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    pred_proba = np.clip(np.asarray(pred, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    groups = np.asarray(patient)
    if not (len(labels) == len(baseline_proba) == len(pred_proba) == len(groups)):
        raise ValueError("y, baseline, pred, and patient must have the same length")
    if len(labels) == 0:
        raise ValueError("cluster permutation requires at least one row")
    unique = np.unique(groups)
    if len(unique) == 0:
        raise ValueError("cluster permutation requires at least one patient cluster")

    baseline_loss = _log_loss_rows(labels, baseline_proba)
    pred_loss = _log_loss_rows(labels, pred_proba)
    row_delta = (baseline_loss - pred_loss) / np.log(2.0)
    cluster_delta = np.asarray(
        [float(np.sum(row_delta[groups == group]) / len(labels)) for group in unique],
        dtype=np.float64,
    )
    observed = float(np.sum(cluster_delta))

    if len(unique) <= 12:
        n_draws = 2 ** len(unique)
        signs = np.empty((n_draws, len(unique)), dtype=np.float64)
        for idx in range(n_draws):
            signs[idx] = [1.0 if (idx >> bit) & 1 else -1.0 for bit in range(len(unique))]
        mode = "exact"
    else:
        n_draws = int(n_permutations)
        if n_draws <= 0:
            raise ValueError("n_permutations must be positive")
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=(n_draws, len(unique)))
        mode = "sampled"
    null = np.sum(signs * cluster_delta[None, :], axis=1)
    exceedances = int(np.sum(null >= observed))
    p_value = float((exceedances + 1) / (len(null) + 1))
    return {
        "method": "patient_cluster_sign_flip_rfs",
        "permutation_unit": "patient_cluster_sign_flip",
        "alternative": "greater",
        "statistic": "rfs_bits",
        "alpha": float(alpha),
        "seed": int(seed),
        "mode": mode,
        "n_clusters": int(len(unique)),
        "n_permutations": int(len(null)),
        "p_resolution": float(1.0 / (len(null) + 1)),
        "observed_rfs_bits": observed,
        "null_mean_rfs_bits": float(np.mean(null)),
        "null_ci_low": float(np.percentile(null, 2.5)),
        "null_ci_high": float(np.percentile(null, 97.5)),
        "p_value": p_value,
        "significant": bool(observed > 0.0 and p_value <= alpha),
        "scope": "primary_horizon_control_only_not_claim_enabling",
    }


def _log_loss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))


def _cluster_permutation_failures(permutation: Any, *, prefix: str) -> list[str]:
    if not isinstance(permutation, dict):
        return [f"{prefix}_cluster_permutation_missing"]
    try:
        p_value = float(permutation["p_value"])
        observed = float(permutation["observed_rfs_bits"])
        alpha = float(permutation.get("alpha", CLUSTER_PERMUTATION_ALPHA))
    except (KeyError, TypeError, ValueError):
        return [f"{prefix}_cluster_permutation_invalid"]
    if not np.isfinite(p_value) or not np.isfinite(observed) or not np.isfinite(alpha):
        return [f"{prefix}_cluster_permutation_nonfinite"]
    if observed <= 0.0 or p_value > alpha:
        return [f"{prefix}_cluster_permutation_not_significant"]
    return []


def _cluster_permutation_is_significant(permutation: Any) -> bool:
    if not isinstance(permutation, dict):
        return False
    try:
        observed = float(permutation["observed_rfs_bits"])
        p_value = float(permutation["p_value"])
        alpha = float(permutation.get("alpha", CLUSTER_PERMUTATION_ALPHA))
        return bool(observed > 0.0 and p_value <= alpha)
    except (KeyError, TypeError, ValueError):
        return False


def _cluster_permutation_is_valid(permutation: Any) -> bool:
    if not isinstance(permutation, dict):
        return False
    try:
        p_value = float(permutation["p_value"])
        observed = float(permutation["observed_rfs_bits"])
        alpha = float(permutation.get("alpha", CLUSTER_PERMUTATION_ALPHA))
    except (KeyError, TypeError, ValueError):
        return False
    return bool(np.isfinite(p_value) and np.isfinite(observed) and np.isfinite(alpha))


def _nuisance_probe_failures(probes: Any, *, prefix: str) -> list[str]:
    if not isinstance(probes, dict):
        probes = {}
    failures: list[str] = []
    for key in NUISANCE_PROBE_KEYS:
        probe = probes.get(key)
        if not isinstance(probe, dict):
            failures.append(f"{prefix}_nuisance_probe_{key}_missing")
            continue
        try:
            accuracy = float(probe["accuracy"])
            chance = float(probe["chance"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{prefix}_nuisance_probe_{key}_invalid")
            continue
        if not np.isfinite(accuracy) or not np.isfinite(chance):
            failures.append(f"{prefix}_nuisance_probe_{key}_nonfinite")
        elif chance < 0.95 and accuracy > chance + NUISANCE_PROBE_MARGIN:
            failures.append(f"{prefix}_nuisance_probe_{key}_above_threshold")
    return failures


def _sleep_curve_failures(
    payload: dict[str, Any],
    *,
    primary_horizon: int = DEFAULT_HORIZONS[0],
) -> list[str]:
    if payload["status"] != "completed_sleep_edf_smoke":
        return ["sleep_edf_smoke_not_completed"]
    first = _primary_row(payload, primary_horizon)
    failures = []
    if first["event_patients"] < 8:
        failures.append("sleep_edf_underpowered_event_patients")
    try:
        positive_events = int(first["positive_events"])
    except (KeyError, TypeError, ValueError):
        failures.append("sleep_edf_primary_positive_events_missing")
    else:
        if positive_events < SLEEP_EDF_MIN_PRIMARY_POSITIVE_EVENTS:
            failures.append("sleep_edf_primary_positive_events_underpowered")
    if first["rfs_ci_low"] <= 0.0:
        failures.append("sleep_edf_primary_rfs_ci_includes_zero")
    if first["shuffled_rfs_bits"] >= first["rfs_bits"] * 0.5:
        failures.append("sleep_edf_shuffled_control_too_close")
    if first["time_shift_rfs_bits"] >= first["rfs_bits"] * 0.5:
        failures.append("sleep_edf_time_shift_control_too_close")
    if first.get("cluster_permutation_failures", ["cluster_permutation_missing"]):
        failures.append("sleep_edf_primary_cluster_permutation_not_significant")
    curve = payload.get("curve", [])
    if not curve:
        failures.append("sleep_edf_nuisance_probe_missing")
    for row in curve:
        horizon = row.get("horizon", "unknown")
        for failure in row.get("nuisance_probe_failures", ["nuisance_probe_missing"]):
            if failure.startswith("horizon_"):
                failures.append(f"sleep_edf_{failure}")
            else:
                failures.append(f"sleep_edf_horizon_{horizon}_{failure}")
    return failures


def _pair_fingerprints(pairs: list[tuple[Path, Path]]) -> list[dict[str, str]]:
    return [
        {
            "record_id": psg.name.removesuffix("-PSG.edf"),
            "psg": psg.name,
            "psg_sha256": _sha256(psg),
            "hypnogram": hyp.name,
            "hypnogram_sha256": _sha256(hyp),
        }
        for psg, hyp in pairs
    ]


def _fingerprint_hash(fingerprints: list[dict[str, str]]) -> str:
    encoded = json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_report(path: Path, gate: dict[str, Any]) -> None:
    lines = [
        "# Kahlus Forecastability Trial 0 - M4 Evidence Report",
        "",
        f"Gate passed: `{gate['gate_passed']}`",
        f"Synthetic gate passed: `{gate['synthetic_gate_passed']}`",
        f"Sleep-EDF smoke status: `{gate['sleep_edf_smoke']['status']}`",
        f"Sleep-EDF smoke failures: `{', '.join(gate['sleep_edf_smoke_failures']) if gate['sleep_edf_smoke_failures'] else 'none'}`",
        "",
        "## Method",
        "",
        "Leakage-safe horizon-wise label curve: labels are shifted within each patient only, "
        "terminal rows without a within-patient future label are excluded before fitting, then "
        "RFS is recomputed per horizon against the strongest gated nuisance/trivial baseline. "
        "This is not a censoring-aware survival model.",
        "",
        "Nuisance probes are reported for every M4 horizon and are claim blockers "
        f"if accuracy exceeds chance + {NUISANCE_PROBE_MARGIN:.2f}; "
        "passing probes do not unlock any clinical, causal, or model-superiority claim.",
        "",
        "M4 also reports patient-cluster sign-flip permutation p-values for horizon RFS. "
        "Only the preregistered primary horizon is inferential for the gate; other horizons "
        "remain descriptive unless a multiplicity-controlled protocol is added.",
        "",
        _curve_section("Synthetic Known Signal", gate["synthetic_known_signal"]),
        "",
        _curve_section("Synthetic Null", gate["synthetic_null"]),
    ]
    if gate["sleep_edf_smoke"]["status"] == "completed_sleep_edf_smoke":
        lines.extend(["", _curve_section("Sleep-EDF Smoke", gate["sleep_edf_smoke"])])
    else:
        lines.extend(["", "## Sleep-EDF Smoke", "", f"- status: `{gate['sleep_edf_smoke']['status']}`"])
    lines.extend(["", "M4 is a benchmark-method gate only; no clinical or foundation-model claim is permitted."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _curve_section(title: str, payload: dict[str, Any]) -> str:
    lines = [
        f"## {title}",
        "",
        "| horizon | total rows | valid rows | evaluated rows | invalid terminal | RFS bits | CI low | CI high | events | "
        "event patients | gated baseline | shuffle | time-shift | "
        "cluster p | nuisance probes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for row in payload["curve"]:
        lines.append(
            "| {horizon} | {total_rows} | {valid_rows} | {evaluated_rows} | "
            "{invalid_terminal_rows} | {rfs_bits:.6f} | {rfs_ci_low:.6f} | "
            "{rfs_ci_high:.6f} | {positive_events} | "
            "{event_patients} | {gated_baseline_name} | {shuffled_rfs_bits:.6f} | "
            "{time_shift_rfs_bits:.6f} | "
            "{cluster_p:.6f} | "
            "{probe_summary} |".format(
                cluster_p=_cluster_p_value(row),
                probe_summary=_format_probe_summary(row),
                **row,
            )
        )
    lines.append("")
    lines.append(f"- positive-RFS AUC: `{payload['auc_positive_rfs_bits']:.6f}` bits")
    return "\n".join(lines)


def _format_probe_summary(row: dict[str, Any]) -> str:
    failures = row.get("nuisance_probe_failures", ["nuisance_probe_missing"])
    if failures:
        return ", ".join(f"`{failure}`" for failure in failures)
    probes = row.get("nuisance_probes", {})
    if not isinstance(probes, dict):
        probes = {}
    values = []
    for key in NUISANCE_PROBE_KEYS:
        probe = probes.get(key, {})
        try:
            values.append(
                f"{key}={float(probe['accuracy']):.3f}/{float(probe['chance']):.3f}"
            )
        except (KeyError, TypeError, ValueError):
            values.append(f"{key}=missing")
    return "passed (" + ", ".join(values) + ")"


def _cluster_p_value(row: dict[str, Any]) -> float:
    try:
        return float(row["cluster_permutation"]["p_value"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
