"""Operator-recovery falsification diagnostics for the v3 Transition Gym.

PROPOSED / SYNTHETIC ONLY. These probe whether the Gym's *hidden* perturbation operators are
identifiable from observed state transitions, whether single-operator estimates generalize to
*unseen* AB/BA compositions, whether the battery is genuinely non-commutative, and how
separable / subject-specific the response profiles are. Everything is leakage-safe (estimators
fit on train episodes, scored on held-out test episodes) and reports honest numbers. No claim of
real brain-state recovery or control is implied.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from neurotwin.falsification import Outcome
from neurotwin.numerics import ignore_spurious_matmul_warnings
from neurotwin.scoring.metrics import r2_score
from neurotwin.transition_gym import TransitionGymBundle
from neurotwin.transition_gym.metrics import mean_commutator_gap

# Outcome type lives in the shared falsification core; alias kept for local readability.
V3Outcome = Outcome


def _pre_states(bundle: TransitionGymBundle) -> np.ndarray:
    """Latent pre-perturbation state z_pre per episode (E, Dz)."""

    return np.asarray(bundle.history_states, dtype=np.float64)[:, -1]


def _affine_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map ``y ≈ x @ W.T + c``; returns (W, c)."""

    aug = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    with ignore_spurious_matmul_warnings():
        theta, *_ = np.linalg.lstsq(aug, y, rcond=None)
    w = theta[:-1].T
    c = theta[-1]
    return w, c


def _affine_predict(x: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    with ignore_spurious_matmul_warnings():
        return x @ w.T + c


def operator_recovery(bundle: TransitionGymBundle, threshold: float = 0.9) -> V3Outcome:
    """Recover each hidden operator ``M_k`` from latent transitions and compare to truth.

    For perturbation a_k the true single-step transition is ``z -> M_k z + b_k`` (computed
    exactly by ``library.apply``). We fit an affine estimate on train episodes, score the
    relative operator error vs the known ``M_k``, and check generalization R² on test episodes.
    """

    library, splits = bundle.library, bundle.splits
    z = _pre_states(bundle)
    train = np.asarray(splits.train_episodes, dtype=int)
    test = np.asarray(splits.test_episodes, dtype=int)

    op_scores: dict[str, float] = {}
    test_r2: list[float] = []
    for name in library.names:
        post = library.apply(name, z)  # exact (M_k z + b_k), latent (noiseless)
        w_hat, c_hat = _affine_fit(z[train], post[train])
        true_m = np.asarray(library.get(name).matrix, dtype=np.float64)
        denom = float(np.linalg.norm(true_m))
        rel_err = float(np.linalg.norm(w_hat - true_m)) / denom if denom > 0 else float(np.linalg.norm(w_hat - true_m))
        op_scores[name] = 1.0 - rel_err
        test_r2.append(r2_score(post[test], _affine_predict(z[test], w_hat, c_hat)))

    mean_recovery = float(np.mean(list(op_scores.values())))
    mean_test_r2 = float(np.mean(test_r2))
    detail = {"mean_recovery_score": mean_recovery, "mean_test_r2": mean_test_r2,
              "per_operator_recovery": op_scores}
    ok = mean_recovery >= threshold and mean_test_r2 >= threshold
    return V3Outcome(
        "operator_recovery", ok, detail,
        "" if ok else f"operators not recovered (recovery={mean_recovery:.3f}, test_r2={mean_test_r2:.3f})",
    )


def heldout_composition_recovery(bundle: TransitionGymBundle, threshold: float = 0.9) -> V3Outcome:
    """Predict held-out AB/BA compositions from single-operator estimates only.

    Single-operator affine estimates are fit on train episodes (never on any composition). For
    each held-out ordered pair (a, b) we predict ``T_b(T_a z)`` by composing the estimates and
    score R² against the true composition on test episodes. This tests the composition law
    ``T_{b∘a} ≈ T_b∘T_a`` on genuinely unseen perturbation sequences.
    """

    library, splits = bundle.library, bundle.splits
    z = _pre_states(bundle)
    train = np.asarray(splits.train_episodes, dtype=int)
    test = np.asarray(splits.test_episodes, dtype=int)

    estimates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in library.names:
        post = library.apply(name, z)
        estimates[name] = _affine_fit(z[train], post[train])

    if not splits.heldout_compositions:
        return V3Outcome("heldout_composition_recovery", False, {"n_heldout": 0},
                         "no held-out compositions to evaluate")

    pair_r2: dict[str, float] = {}
    for first, second in splits.heldout_compositions:
        w_a, c_a = estimates[first]
        w_b, c_b = estimates[second]
        pred = _affine_predict(_affine_predict(z[test], w_a, c_a), w_b, c_b)
        true = library.compose(first, second, z[test])  # apply first then second
        pair_r2[f"{first}->{second}"] = r2_score(true, pred)

    mean_r2 = float(np.mean(list(pair_r2.values())))
    detail = {"mean_heldout_composition_r2": mean_r2, "n_heldout": len(pair_r2),
              "per_pair_r2": pair_r2}
    ok = mean_r2 >= threshold
    return V3Outcome(
        "heldout_composition_recovery", ok, detail,
        "" if ok else f"held-out compositions not predicted (mean r2={mean_r2:.3f})",
    )


def non_commutativity_score(bundle: TransitionGymBundle, eps: float = 0.01) -> V3Outcome:
    """Explicit AB-vs-BA gap; the battery must be genuinely non-commutative."""

    library, splits = bundle.library, bundle.splits
    z = _pre_states(bundle)
    all_pairs = list(permutations(library.names, 2))
    response_scale = float(np.mean([np.linalg.norm(library.apply(n, z)) / max(1, z.shape[0]) for n in library.names]))
    response_scale = response_scale if response_scale > 0 else 1.0

    all_gap = mean_commutator_gap(library, all_pairs, z)
    heldout_gap = mean_commutator_gap(library, splits.heldout_compositions, z)
    per_pair = {f"{a}<->{b}": library.commutator_gap(a, b, z) for a, b in all_pairs}
    normalized = all_gap / response_scale
    detail = {"mean_gap_all_pairs": all_gap, "mean_gap_heldout": heldout_gap,
              "normalized_gap": normalized, "per_pair_gap": per_pair}
    ok = normalized > eps
    return V3Outcome(
        "non_commutativity", ok, detail,
        "" if ok else f"battery is (near) commutative (normalized gap={normalized:.4f})",
    )


def response_profile_distances(bundle: TransitionGymBundle, eps: float = 1e-6) -> V3Outcome:
    """Mean trajectory, operator-induced, and subject-transfer response-profile distances."""

    response = np.asarray(bundle.response_eeg, dtype=np.float64)  # (E, K, H, C)
    n_episodes, n_pert = response.shape[0], response.shape[1]
    flat = response.reshape(n_episodes, n_pert, -1)  # (E, K, H*C)

    # Mean trajectory distance: average cross-episode spread within each perturbation.
    traj_dists = []
    for k in range(n_pert):
        prof = flat[:, k]  # (E, H*C)
        centered = prof - prof.mean(axis=0, keepdims=True)
        traj_dists.append(float(np.mean(np.linalg.norm(centered, axis=1))))
    mean_trajectory_distance = float(np.mean(traj_dists))

    # Operator-induced response distance: separability across the K operators' mean profiles.
    op_means = flat.mean(axis=0)  # (K, H*C)
    op_pair = [float(np.linalg.norm(op_means[i] - op_means[j]))
               for i in range(n_pert) for j in range(i + 1, n_pert)]
    operator_induced_distance = float(np.mean(op_pair)) if op_pair else 0.0

    # Subject-transfer distance: same-perturbation profile distance across subjects.
    subject_ids = np.asarray(bundle.world.subject_ids)
    subjects = np.unique(subject_ids)
    subj_dists = []
    if subjects.size >= 2:
        for k in range(n_pert):
            per_subject = [flat[subject_ids == s, k].mean(axis=0) for s in subjects
                           if np.any(subject_ids == s)]
            for i in range(len(per_subject)):
                for j in range(i + 1, len(per_subject)):
                    subj_dists.append(float(np.linalg.norm(per_subject[i] - per_subject[j])))
    subject_transfer_distance = float(np.mean(subj_dists)) if subj_dists else 0.0

    detail = {"mean_trajectory_distance": mean_trajectory_distance,
              "operator_induced_distance": operator_induced_distance,
              "subject_transfer_distance": subject_transfer_distance}
    finite = all(np.isfinite(v) for v in detail.values())
    ok = finite and operator_induced_distance > eps
    return V3Outcome(
        "response_profile_distances", ok, detail,
        "" if ok else "operators not separable by response profile or non-finite distance",
    )
