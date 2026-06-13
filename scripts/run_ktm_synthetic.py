#!/usr/bin/env python3
"""Smoke runner for the Kahlus v3 KTM scaffold on the synthetic Transition Gym.

PROPOSED / SYNTHETIC ONLY. Instantiates a KTM, predicts response profiles / uncertainty on
gym history, reports commutativity of the latent operators, and writes a summary + a
claim-blocking evidence gate. No training, no A100, no scientific claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _bootstrap import ensure_src_import_path

ensure_src_import_path(__file__)

from neurotwin.gates import evaluate_gate, write_evidence_gate  # noqa: E402
from neurotwin.models.ktm import KTM, KTMConfig  # noqa: E402
from neurotwin.repro import write_json  # noqa: E402
from neurotwin.transition_gym import SyntheticWorldConfig, build_transition_gym  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-episodes", type=int, default=48)
    args = parser.parse_args()

    world_cfg = SyntheticWorldConfig(seed=args.seed, n_episodes=args.n_episodes)
    bundle = build_transition_gym(world_cfg)
    ktm = KTM(
        KTMConfig(
            seed=args.seed,
            history_len=world_cfg.history_len,
            eeg_channels=world_cfg.eeg_channels,
            n_perturbations=world_cfg.n_perturbations,
            horizon=world_cfg.horizon,
        )
    )

    history = bundle.history_eeg
    profile = ktm.predict_response_profile(history)
    uncertainty = ktm.predict_uncertainty(history)
    metadata = ktm.metadata()

    finite = bool(np.isfinite(profile).all() and np.isfinite(uncertainty).all())
    out = Path(args.out_dir)
    summary_path = write_json(
        out / "ktm_summary.json",
        {
            "branch": "v3",
            "claim_status": "synthetic_scaffold_only",
            "seed": int(args.seed),
            "response_profile_shape": list(profile.shape),
            "uncertainty_shape": list(uncertainty.shape),
            "uncertainty_min": float(np.min(uncertainty)),
            "max_commutator_norm": metadata["max_commutator_norm"],
            "non_commutative": metadata["non_commutative"],
            "outputs_finite": finite,
        },
    )
    gate = evaluate_gate(
        branch="v3",
        dataset="ktm_synthetic",
        split_audit_passed=True,
        baseline_table_present=False,  # KTM smoke is not a baseline sweep
        finite_metrics=finite,
        calibration_checked=False,
        claim_scope="synthetic_ktm_recovery",
        extra_failure_reasons=["ktm smoke is a forward-pass shape/finite check, not a trained result"],
    )
    gate_path = write_evidence_gate(out / "evidence_gate.json", gate)

    print(f"branch=v3 model=KTM out_dir={out.resolve()}")
    print(f"response_profile_shape={list(profile.shape)} uncertainty_min={float(np.min(uncertainty)):.4g}")
    print(f"non_commutative={metadata['non_commutative']} max_commutator_norm={metadata['max_commutator_norm']:.4f}")
    print(f"outputs_finite={finite} scientific_claim_allowed={gate['scientific_claim_allowed']}")
    print(f"summary={summary_path} evidence_gate={gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
