from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurotwin.forecastability.m1 import TransitionFixture, _run_fixture, make_transition_fixture
from neurotwin.forecastability.m2 import _as_transition_fixture, load_sleep_edf_fixture


DEFAULT_HORIZONS = (1, 2, 3)


def run_m4_gate(
    out_dir: str | Path,
    *,
    seed: int = 0,
    sleep_edf_root: str | Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    known = _curve_payload(make_transition_fixture(seed=seed, residual_signal=True), horizons=horizons, seed=seed)
    null = _curve_payload(make_transition_fixture(seed=seed + 100, residual_signal=False), horizons=horizons, seed=seed + 100)
    sleep = _sleep_edf_curve_payload(sleep_edf_root, horizons=horizons, seed=seed + 200)
    synthetic_passed = _known_curve_passes(known) and _null_curve_passes(null)
    sleep_failures = _sleep_curve_failures(sleep)
    gate = {
        "milestone": "M4",
        "method": "leakage_safe_forecastability_vs_horizon_curve",
        "horizons": list(horizons),
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


def _curve_payload(fixture: TransitionFixture, *, horizons: tuple[int, ...], seed: int) -> dict[str, Any]:
    labels_by_horizon = patient_horizon_labels(fixture.y, fixture.patient, horizons=horizons)
    rows = []
    for offset, horizon in enumerate(horizons):
        payload = _run_fixture(_with_labels(fixture, labels_by_horizon[horizon]), seed=seed + offset)
        rows.append(
            {
                "horizon": horizon,
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
        return {"status": "completed_sleep_edf_smoke", "root": str(root), **payload}
    except Exception as exc:  # noqa: BLE001 - parser/runtime failures are evidence.
        return {"status": "sleep_edf_smoke_failed", "root": str(root), "error": str(exc)}


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


def _known_curve_passes(payload: dict[str, Any]) -> bool:
    first = payload["curve"][0]
    return bool(
        first["positive_events"] >= 40
        and first["event_patients"] >= 8
        and first["rfs_bits"] > 0.03
        and first["rfs_ci_low"] > 0.0
        and first["shuffled_rfs_bits"] < first["rfs_bits"] * 0.5
        and first["time_shift_rfs_bits"] < first["rfs_bits"] * 0.5
    )


def _null_curve_passes(payload: dict[str, Any]) -> bool:
    return bool(all(abs(row["rfs_bits"]) < 0.03 and row["rfs_ci_low"] <= 0.0 <= row["rfs_ci_high"] for row in payload["curve"]))


def _sleep_curve_failures(payload: dict[str, Any]) -> list[str]:
    if payload["status"] != "completed_sleep_edf_smoke":
        return ["sleep_edf_smoke_not_completed"]
    first = payload["curve"][0]
    failures = []
    if first["event_patients"] < 8:
        failures.append("sleep_edf_underpowered_event_patients")
    if first["rfs_ci_low"] <= 0.0:
        failures.append("sleep_edf_primary_rfs_ci_includes_zero")
    if first["shuffled_rfs_bits"] >= first["rfs_bits"] * 0.5:
        failures.append("sleep_edf_shuffled_control_too_close")
    if first["time_shift_rfs_bits"] >= first["rfs_bits"] * 0.5:
        failures.append("sleep_edf_time_shift_control_too_close")
    return failures


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
        "Leakage-safe forecastability-vs-horizon curve: labels are shifted within each patient only, then RFS is recomputed per horizon against the strongest gated nuisance/trivial baseline.",
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
        "| horizon | RFS bits | CI low | CI high | events | event patients | gated baseline | shuffle | time-shift |",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in payload["curve"]:
        lines.append(
            "| {horizon} | {rfs_bits:.6f} | {rfs_ci_low:.6f} | {rfs_ci_high:.6f} | {positive_events} | {event_patients} | {gated_baseline_name} | {shuffled_rfs_bits:.6f} | {time_shift_rfs_bits:.6f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"- positive-RFS AUC: `{payload['auc_positive_rfs_bits']:.6f}` bits")
    return "\n".join(lines)
