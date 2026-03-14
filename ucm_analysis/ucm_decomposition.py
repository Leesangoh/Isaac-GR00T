"""Core UCM decomposition algorithm."""

import numpy as np
from scipy.linalg import null_space


def compute_ucm_decomposition(errors, J):
    """
    UCM decomposition of error vectors.

    Args:
        errors: (N, D) array of error vectors
        J: (task_dim, D) Jacobian mapping action space to task variable

    Returns:
        dict with V_ucm, V_ort (variance per DOF), ratio, dimensions
    """
    N, D = errors.shape

    # Null space of J = UCM
    UCM_basis = null_space(J)  # (D, d_ucm)
    d_ucm = UCM_basis.shape[1]

    # Orthogonal complement = range space of J.T
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    rank = np.sum(S > 1e-10)
    ORT_basis = Vt[:rank].T  # (D, d_ort)
    d_ort = ORT_basis.shape[1]

    # Project errors
    proj_ucm = errors @ UCM_basis  # (N, d_ucm)
    proj_ort = errors @ ORT_basis  # (N, d_ort)

    # Variance per DOF
    V_ucm = np.sum(proj_ucm ** 2) / (N * d_ucm) if d_ucm > 0 else 0.0
    V_ort = np.sum(proj_ort ** 2) / (N * d_ort) if d_ort > 0 else 0.0

    ratio = V_ucm / V_ort if V_ort > 0 else float('inf')

    return {
        "V_ucm": V_ucm,
        "V_ort": V_ort,
        "ratio": ratio,
        "d_ucm": d_ucm,
        "d_ort": d_ort,
        "proj_ucm_norms": np.linalg.norm(proj_ucm, axis=1),
        "proj_ort_norms": np.linalg.norm(proj_ort, axis=1),
    }


def compute_ucm_per_episode(episode_errors_list, J):
    """
    Compute UCM ratio per episode for statistical testing.

    Args:
        episode_errors_list: list of (T_i, D) arrays, one per episode
        J: Jacobian

    Returns:
        list of per-episode UCM ratios
    """
    UCM_basis = null_space(J)
    d_ucm = UCM_basis.shape[1]

    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    rank = np.sum(S > 1e-10)
    ORT_basis = Vt[:rank].T
    d_ort = ORT_basis.shape[1]

    ratios = []
    for errors in episode_errors_list:
        if len(errors) < 2:
            continue
        N = len(errors)
        proj_ucm = errors @ UCM_basis
        proj_ort = errors @ ORT_basis
        v_ucm = np.sum(proj_ucm ** 2) / (N * d_ucm) if d_ucm > 0 else 0.0
        v_ort = np.sum(proj_ort ** 2) / (N * d_ort) if d_ort > 0 else 0.0
        if v_ort > 0:
            ratios.append(v_ucm / v_ort)
    return np.array(ratios)
