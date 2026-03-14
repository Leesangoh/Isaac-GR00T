"""Visualization utilities for UCM analysis."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os


def plot_ucm_ratio_bar(results_dict, output_path, title="UCM Ratio by Task Variable"):
    """Bar chart of V_UCM/V_ORT ratio for each task variable."""
    fig, ax = plt.subplots(figsize=(8, 5))

    names = list(results_dict.keys())
    ratios = [results_dict[n]["ratio"] for n in names]
    v_ucm = [results_dict[n]["V_ucm"] for n in names]
    v_ort = [results_dict[n]["V_ort"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, v_ucm, width, label='V_UCM/dof', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, v_ort, width, label='V_ORT/dof', color='coral', alpha=0.8)

    ax.set_ylabel('Variance per DOF')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names], fontsize=9)
    ax.legend()

    # Add ratio text above bars
    for i, r in enumerate(ratios):
        y_max = max(v_ucm[i], v_ort[i])
        ax.text(i, y_max * 1.05, f'ratio={r:.2f}', ha='center', fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_ucm_ratio_distribution(episode_ratios_dict, output_path):
    """Histogram of per-episode UCM ratios for each task variable."""
    n_tv = len(episode_ratios_dict)
    fig, axes = plt.subplots(1, n_tv, figsize=(5 * n_tv, 4), squeeze=False)

    for i, (name, ratios) in enumerate(episode_ratios_dict.items()):
        ax = axes[0, i]
        # Clip extreme ratios for visualization
        clipped = np.clip(ratios, 0, np.percentile(ratios, 99))
        ax.hist(clipped, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='ratio=1 (null)')
        ax.axvline(x=np.median(ratios), color='orange', linestyle='-', linewidth=2, label=f'median={np.median(ratios):.2f}')
        ax.set_xlabel('V_UCM / V_ORT')
        ax.set_ylabel('Count')
        ax.set_title(name.replace('_', ' '))
        ax.legend(fontsize=8)

    plt.suptitle('Per-Episode UCM Ratio Distribution', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_phase_ucm(phase_results, output_path):
    """UCM ratio by task phase for each task variable."""
    from ucm_analysis.phase_segmentation import PHASE_NAMES

    tv_names = list(list(phase_results.values())[0].keys())
    phases = sorted(phase_results.keys())
    phase_labels = [PHASE_NAMES.get(p, str(p)) for p in phases]

    fig, axes = plt.subplots(1, len(tv_names), figsize=(5 * len(tv_names), 5), squeeze=False)

    for j, tv in enumerate(tv_names):
        ax = axes[0, j]
        ratios = [phase_results[p][tv]["ratio"] for p in phases]
        v_ucm = [phase_results[p][tv]["V_ucm"] for p in phases]
        v_ort = [phase_results[p][tv]["V_ort"] for p in phases]

        x = np.arange(len(phases))
        width = 0.35
        ax.bar(x - width/2, v_ucm, width, label='V_UCM/dof', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, v_ort, width, label='V_ORT/dof', color='coral', alpha=0.8)

        for i, r in enumerate(ratios):
            y_max = max(v_ucm[i], v_ort[i])
            ax.text(i, y_max * 1.05, f'{r:.2f}', ha='center', fontsize=9, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(phase_labels, fontsize=9, rotation=30)
        ax.set_ylabel('Variance per DOF')
        ax.set_title(tv.replace('_', ' '))
        ax.legend(fontsize=8)
        ax.axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle('UCM Ratio by Task Phase', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_chunk_step_ucm(step_results, output_path):
    """UCM ratio across chunk steps 0-7."""
    tv_names = list(list(step_results.values())[0].keys())
    steps = sorted(step_results.keys())

    fig, axes = plt.subplots(1, len(tv_names), figsize=(5 * len(tv_names), 4), squeeze=False)

    for j, tv in enumerate(tv_names):
        ax = axes[0, j]
        ratios = [step_results[s][tv]["ratio"] for s in steps]
        v_ucm = [step_results[s][tv]["V_ucm"] for s in steps]
        v_ort = [step_results[s][tv]["V_ort"] for s in steps]

        ax.plot(steps, ratios, 'ko-', linewidth=2, markersize=6, label='V_UCM/V_ORT')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='ratio=1')
        ax.set_xlabel('Chunk Step')
        ax.set_ylabel('UCM Ratio')
        ax.set_title(tv.replace('_', ' '))
        ax.legend(fontsize=8)

    plt.suptitle('UCM Ratio by Chunk Step', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_projection_scatter(results_dict, output_path):
    """Scatter plot of ||proj_ORT|| vs ||proj_UCM|| for each TV."""
    n_tv = len(results_dict)
    fig, axes = plt.subplots(1, n_tv, figsize=(5 * n_tv, 5), squeeze=False)

    for i, (name, res) in enumerate(results_dict.items()):
        ax = axes[0, i]
        ucm_norms = res["proj_ucm_norms"]
        ort_norms = res["proj_ort_norms"]

        # Subsample for visualization if too many points
        n = len(ucm_norms)
        if n > 5000:
            idx = np.random.choice(n, 5000, replace=False)
            ucm_norms = ucm_norms[idx]
            ort_norms = ort_norms[idx]

        ax.scatter(ort_norms, ucm_norms, alpha=0.1, s=3, c='steelblue')
        ax.set_xlabel('||proj_ORT(error)||')
        ax.set_ylabel('||proj_UCM(error)||')
        ax.set_title(name.replace('_', ' '))

        # Add diagonal
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax.legend(fontsize=8)

    plt.suptitle('Error Projections: UCM vs ORT', fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_variance_eigenspectrum(errors, J, output_path):
    """Eigenvalue spectrum of error covariance, annotated with UCM/ORT."""
    from scipy.linalg import null_space

    cov = np.cov(errors.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Compute alignment of each eigenvector with UCM
    UCM_basis = null_space(J)
    ucm_alignment = np.zeros(len(eigenvalues))
    for i in range(len(eigenvalues)):
        proj = UCM_basis.T @ eigenvectors[:, i]
        ucm_alignment[i] = np.linalg.norm(proj)  # 0=pure ORT, 1=pure UCM

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Eigenspectrum
    colors = plt.cm.RdYlBu(ucm_alignment)
    ax1.bar(range(len(eigenvalues)), eigenvalues, color=colors, edgecolor='none')
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Error Covariance Eigenspectrum (color = UCM alignment)')
    sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=ax1, label='UCM alignment (1=UCM, 0=ORT)')

    # Cumulative variance explained
    cum_var = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    ax2.plot(range(len(eigenvalues)), cum_var, 'k-', linewidth=2)
    ax2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='90%')
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Variance Explained')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return eigenvalues, ucm_alignment
