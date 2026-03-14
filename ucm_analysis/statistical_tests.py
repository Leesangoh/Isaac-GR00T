"""Statistical tests for UCM analysis."""

import numpy as np
from scipy import stats


def bootstrap_ucm_ratio(episode_ratios, n_bootstrap=10000, ci=95):
    """
    Bootstrap confidence interval for mean UCM ratio.

    Args:
        episode_ratios: array of per-episode UCM ratios
        n_bootstrap: number of bootstrap samples
        ci: confidence interval percentage

    Returns:
        dict with mean, median, CI bounds, p-value (H0: ratio=1)
    """
    n = len(episode_ratios)
    if n < 2:
        return {"mean": np.nan, "median": np.nan, "ci_low": np.nan,
                "ci_high": np.nan, "p_value": np.nan, "n": n}

    boot_means = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        sample = np.random.choice(episode_ratios, size=n, replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_means, alpha)
    ci_high = np.percentile(boot_means, 100 - alpha)

    # One-sample test against ratio=1
    # Using log-transform for ratio data
    log_ratios = np.log(episode_ratios[episode_ratios > 0])
    if len(log_ratios) > 1:
        t_stat, p_value = stats.ttest_1samp(log_ratios, 0)  # H0: log(ratio) = 0 => ratio = 1
    else:
        p_value = np.nan

    return {
        "mean": np.mean(episode_ratios),
        "median": np.median(episode_ratios),
        "std": np.std(episode_ratios),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
        "n": n,
    }


def permutation_test_ucm(errors, J, n_permutations=5000):
    """
    Permutation test: shuffle error components across dimensions to break
    UCM structure, compare observed ratio to permuted distribution.
    """
    from ucm_analysis.ucm_decomposition import compute_ucm_decomposition

    observed = compute_ucm_decomposition(errors, J)
    observed_ratio = observed["ratio"]

    N, D = errors.shape
    perm_ratios = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle each error vector's components independently
        shuffled = errors.copy()
        for n in range(N):
            np.random.shuffle(shuffled[n])
        result = compute_ucm_decomposition(shuffled, J)
        perm_ratios[i] = result["ratio"]

    p_value = np.mean(perm_ratios >= observed_ratio)

    return {
        "observed_ratio": observed_ratio,
        "perm_mean": np.mean(perm_ratios),
        "perm_std": np.std(perm_ratios),
        "p_value": p_value,
    }
