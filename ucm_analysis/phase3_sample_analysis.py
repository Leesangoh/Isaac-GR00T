"""
Phase 3: Multi-Sample UCM Analysis
====================================
Analyzes the variability structure of K action samples from the same observation.

Key difference from Phase 1:
  - Phase 1: VLA-Expert error structure across different timesteps
  - Phase 3: VLA internal variability structure at the same timestep

Usage:
  python -m ucm_analysis.phase3_sample_analysis \
    --data_dir ucm_analysis/results/multi_samples \
    --output_dir ucm_analysis/results/phase3
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucm_analysis.config import N_STEPS, ACTION_DIM, ACTION_LABELS
from ucm_analysis.task_variables import TASK_VARIABLES_NO_GRIPPER
from ucm_analysis.ucm_decomposition import compute_ucm_decomposition
from ucm_analysis.phase_segmentation import segment_episode_phases, PHASE_NAMES
from ucm_analysis.statistical_tests import bootstrap_ucm_ratio
from ucm_analysis.visualization import (
    plot_ucm_ratio_bar,
    plot_ucm_ratio_distribution,
    plot_phase_ucm,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_multi_sample_data(data_dir):
    """Load all multi-sample episode files."""
    files = sorted(glob(os.path.join(data_dir, "episode_*.npz")))
    episodes = []
    for f in files:
        data = np.load(f)
        episodes.append({
            "file": f,
            "samples": data["samples"],          # (T, K, 8, 7)
            "expert": data["expert"],             # (T, 8, 7)
            "timestep_ids": data["timestep_ids"],  # (T,)
            "episode_idx": int(data["episode_idx"]),
            "traj_length": int(data["traj_length"]),
        })
    return episodes


def analyze_per_timestep_ucm(episodes, include_gripper=False):
    """
    UCM decomposition of K samples at each timestep.
    Returns per-timestep results as DataFrame.
    """
    tv_dict = TASK_VARIABLES_NO_GRIPPER
    results = []

    for ep in episodes:
        samples = ep["samples"]   # (T, K, 8, 7)
        expert = ep["expert"]     # (T, 8, 7)
        T, K, n_steps, action_dim = samples.shape

        for t in range(T):
            # Remove gripper
            if include_gripper:
                action_vecs = samples[t].reshape(K, n_steps * action_dim)  # (K, 56)
            else:
                action_vecs = samples[t, :, :, :6].reshape(K, n_steps * 6)  # (K, 48)

            # Mean-center (deviations from mean prediction)
            mean_action = action_vecs.mean(axis=0)
            deviations = action_vecs - mean_action  # (K, D)

            # Skip if no variance
            if np.sum(deviations ** 2) < 1e-12:
                continue

            row = {
                "episode_idx": ep["episode_idx"],
                "timestep": int(ep["timestep_ids"][t]),
                "traj_length": ep["traj_length"],
                "K": K,
                "total_var": np.sum(np.var(action_vecs, axis=0)),
                "expert_action_norm": np.linalg.norm(expert[t, 0, :6]),
                "gripper_state": expert[t, 0, 6],
            }

            # UCM decomposition for each task variable
            for tv_name, build_J in tv_dict.items():
                J = build_J(n_steps=n_steps, action_dim=6)
                result = compute_ucm_decomposition(deviations, J)
                row[f"{tv_name}_V_ucm"] = result["V_ucm"]
                row[f"{tv_name}_V_ort"] = result["V_ort"]
                row[f"{tv_name}_ratio"] = result["ratio"]

            results.append(row)

    return pd.DataFrame(results)


def analyze_expert_direction(episodes):
    """
    For each timestep: variance along expert action direction vs orthogonal.
    Tests whether VLA variability is aligned with expert intention.
    """
    results = []
    for ep in episodes:
        samples = ep["samples"]
        expert = ep["expert"]
        T, K, n_steps, action_dim = samples.shape

        for t in range(T):
            # Flatten to 48D (no gripper)
            action_vecs = samples[t, :, :, :6].reshape(K, n_steps * 6)
            expert_vec = expert[t, :, :6].flatten()

            expert_norm = np.linalg.norm(expert_vec)
            if expert_norm < 1e-8:
                continue

            expert_dir = expert_vec / expert_norm
            mean_action = action_vecs.mean(axis=0)
            deviations = action_vecs - mean_action

            # Variance along expert direction
            proj_along = deviations @ expert_dir  # (K,)
            var_along = np.var(proj_along)

            # Variance in orthogonal complement
            total_var = np.sum(np.var(deviations, axis=0))
            var_ortho = (total_var - var_along) / (deviations.shape[1] - 1)

            results.append({
                "episode_idx": ep["episode_idx"],
                "timestep": int(ep["timestep_ids"][t]),
                "var_along_expert": var_along,
                "var_ortho_expert": var_ortho,
                "ratio_along_ortho": var_along / var_ortho if var_ortho > 0 else float('inf'),
                "expert_action_norm": expert_norm,
            })

    return pd.DataFrame(results)


def analyze_pca_ucm_alignment(episodes):
    """
    PCA of K samples, check if top PCs align with UCM or ORT.
    """
    from scipy.linalg import null_space

    tv_dict = TASK_VARIABLES_NO_GRIPPER
    results = []

    for ep in episodes:
        samples = ep["samples"]
        T, K, n_steps, action_dim = samples.shape

        for t in range(T):
            action_vecs = samples[t, :, :, :6].reshape(K, n_steps * 6)
            mean_action = action_vecs.mean(axis=0)
            deviations = action_vecs - mean_action

            if np.sum(deviations ** 2) < 1e-12:
                continue

            # PCA
            cov = (deviations.T @ deviations) / K
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            eigenvalues = eigenvalues[::-1]
            eigenvectors = eigenvectors[:, ::-1]

            # Top-5 PCs
            top_k = min(5, len(eigenvalues))
            total_var = np.sum(eigenvalues)

            for tv_name, build_J in tv_dict.items():
                J = build_J(n_steps=n_steps, action_dim=6)
                UCM_basis = null_space(J)

                for pc_idx in range(top_k):
                    # UCM alignment: how much of this PC lies in UCM
                    proj = UCM_basis.T @ eigenvectors[:, pc_idx]
                    ucm_alignment = np.linalg.norm(proj)

                    results.append({
                        "episode_idx": ep["episode_idx"],
                        "timestep": int(ep["timestep_ids"][t]),
                        "tv_name": tv_name,
                        "pc_index": pc_idx,
                        "eigenvalue": eigenvalues[pc_idx],
                        "var_explained": eigenvalues[pc_idx] / total_var if total_var > 0 else 0,
                        "ucm_alignment": ucm_alignment,
                    })

    return pd.DataFrame(results)


def plot_phase3_results(df, expert_dir_df, pca_df, output_dir):
    """Generate Phase 3 specific plots."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    tv_names = [c.replace("_ratio", "") for c in df.columns if c.endswith("_ratio")]

    # 1. Per-timestep UCM ratio distributions
    fig, axes = plt.subplots(1, len(tv_names), figsize=(5 * len(tv_names), 4), squeeze=False)
    for i, tv in enumerate(tv_names):
        ax = axes[0, i]
        ratios = df[f"{tv}_ratio"].dropna()
        ratios_clipped = np.clip(ratios, 0, np.percentile(ratios, 99))
        ax.hist(ratios_clipped, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='ratio=1')
        ax.axvline(x=ratios.median(), color='orange', linewidth=2,
                   label=f'median={ratios.median():.2f}')
        ax.set_xlabel('V_UCM / V_ORT')
        ax.set_ylabel('Count')
        ax.set_title(tv.replace('_', ' '))
        ax.legend(fontsize=8)
    plt.suptitle('Phase 3: Per-Timestep UCM Ratio (Multi-Sample)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "phase3_ucm_ratio_dist.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. UCM ratio vs expert action magnitude
    fig, axes = plt.subplots(1, len(tv_names), figsize=(5 * len(tv_names), 4), squeeze=False)
    for i, tv in enumerate(tv_names):
        ax = axes[0, i]
        valid = df[["expert_action_norm", f"{tv}_ratio"]].dropna()
        if len(valid) > 2000:
            valid = valid.sample(2000, random_state=42)
        ax.scatter(valid["expert_action_norm"], valid[f"{tv}_ratio"],
                   alpha=0.2, s=5, c='steelblue')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('||Expert Action||')
        ax.set_ylabel('UCM Ratio')
        ax.set_title(tv.replace('_', ' '))
        ax.set_ylim(0, np.percentile(valid[f"{tv}_ratio"], 98))
    plt.suptitle('UCM Ratio vs Expert Action Magnitude', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "phase3_ratio_vs_action_norm.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Expert direction alignment
    if len(expert_dir_df) > 0:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ratios = expert_dir_df["ratio_along_ortho"].dropna()
        ratios_clipped = np.clip(ratios, 0, np.percentile(ratios, 99))
        ax1.hist(ratios_clipped, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
        ax1.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='ratio=1')
        ax1.axvline(x=ratios.median(), color='orange', linewidth=2,
                    label=f'median={ratios.median():.2f}')
        ax1.set_xlabel('Var(along expert) / Var(ortho expert)')
        ax1.set_ylabel('Count')
        ax1.set_title('Variability Along vs Orthogonal to Expert Direction')
        ax1.legend()

        # Scatter: var_along vs var_ortho
        if len(expert_dir_df) > 2000:
            sub = expert_dir_df.sample(2000, random_state=42)
        else:
            sub = expert_dir_df
        ax2.scatter(sub["var_ortho_expert"], sub["var_along_expert"],
                    alpha=0.2, s=5, c='steelblue')
        max_val = max(ax2.get_xlim()[1], ax2.get_ylim()[1])
        ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
        ax2.set_xlabel('Var orthogonal to expert')
        ax2.set_ylabel('Var along expert')
        ax2.set_title('Expert Direction Variance Decomposition')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "phase3_expert_direction.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 4. PCA-UCM alignment
    if len(pca_df) > 0:
        fig, axes = plt.subplots(1, len(tv_names), figsize=(5 * len(tv_names), 4), squeeze=False)
        for i, tv in enumerate(tv_names):
            ax = axes[0, i]
            sub = pca_df[pca_df["tv_name"] == tv]
            for pc in range(min(5, sub["pc_index"].max() + 1)):
                pc_data = sub[sub["pc_index"] == pc]
                ax.hist(pc_data["ucm_alignment"], bins=30, alpha=0.5,
                        label=f'PC{pc} (var={pc_data["var_explained"].mean():.2f})')
            ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('UCM Alignment (0=ORT, 1=UCM)')
            ax.set_ylabel('Count')
            ax.set_title(tv.replace('_', ' '))
            ax.legend(fontsize=7)
        plt.suptitle('PCA Components: UCM Alignment', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "phase3_pca_ucm_alignment.png"), dpi=150, bbox_inches='tight')
        plt.close()

    # 5. Total variance across episodes
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["total_var"].dropna(), bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Total Variance (sum of per-dim variances)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of VLA Action Variability Across Timesteps')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "phase3_total_variance.png"), dpi=150, bbox_inches='tight')
    plt.close()


def generate_phase3_report(df, expert_dir_df, pca_df, output_dir):
    """Generate Phase 3 markdown report."""
    tv_names = [c.replace("_ratio", "") for c in df.columns if c.endswith("_ratio")]

    report_path = os.path.join(output_dir, "phase3_results.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 3: Multi-Sample UCM Analysis\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis examines the **internal variability** of VLA predictions\n")
        f.write("by running GR00T N1.6 K times per observation with different random seeds.\n\n")
        f.write("**Key difference from Phase 1**:\n")
        f.write("- Phase 1: Structure of VLA-Expert *error* across timesteps\n")
        f.write("- Phase 3: Structure of VLA *internal variability* at each timestep\n\n")

        n_episodes = df["episode_idx"].nunique()
        n_timesteps = len(df)
        K = df["K"].iloc[0] if len(df) > 0 else 0
        f.write(f"**Data**: {n_episodes} episodes, {n_timesteps} timesteps, K={K} samples each\n\n")

        # 1. Global UCM results
        f.write("## 1. UCM Decomposition of Multi-Sample Variability\n\n")
        f.write("| Task Variable | Median Ratio | Mean Ratio | % with ratio>1 | Mean V_UCM/dof | Mean V_ORT/dof |\n")
        f.write("|---|---|---|---|---|---|\n")
        for tv in tv_names:
            ratios = df[f"{tv}_ratio"].dropna()
            v_ucm = df[f"{tv}_V_ucm"].dropna()
            v_ort = df[f"{tv}_V_ort"].dropna()
            pct_above_1 = (ratios > 1).mean() * 100
            f.write(f"| {tv} | **{ratios.median():.4f}** | {ratios.mean():.4f} | "
                    f"{pct_above_1:.1f}% | {v_ucm.mean():.6f} | {v_ort.mean():.6f} |\n")

        # Bootstrap CIs
        f.write("\n### Bootstrap Statistics\n\n")
        f.write("| Task Variable | Mean | 95% CI | p-value (H0: ratio=1) |\n")
        f.write("|---|---|---|---|\n")
        for tv in tv_names:
            ratios = df[f"{tv}_ratio"].dropna().values
            boot = bootstrap_ucm_ratio(ratios)
            f.write(f"| {tv} | {boot['mean']:.4f} | [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] | "
                    f"{boot['p_value']:.2e} |\n")

        # 2. Expert direction analysis
        if len(expert_dir_df) > 0:
            f.write("\n## 2. Expert Direction Alignment\n\n")
            ratios = expert_dir_df["ratio_along_ortho"].dropna()
            f.write(f"- **Median** var(along expert) / var(ortho): **{ratios.median():.4f}**\n")
            f.write(f"- **Mean**: {ratios.mean():.4f}\n")
            f.write(f"- **% with ratio > 1**: {(ratios > 1).mean() * 100:.1f}%\n\n")
            if ratios.median() < 1:
                f.write("Interpretation: VLA variability is *lower* along expert action direction\n")
                f.write("→ VLA is relatively more certain about the expert's intended direction.\n\n")
            else:
                f.write("Interpretation: VLA variability is *higher* along expert action direction\n")
                f.write("→ VLA is less certain about the magnitude of the expert's intended action.\n\n")

        # 3. PCA-UCM alignment
        if len(pca_df) > 0:
            f.write("## 3. PCA-UCM Alignment\n\n")
            f.write("Top principal components of VLA variability and their UCM alignment:\n\n")
            f.write("| Task Variable | PC | Mean Var Explained | Mean UCM Alignment |\n")
            f.write("|---|---|---|---|\n")
            for tv in tv_names:
                sub = pca_df[pca_df["tv_name"] == tv]
                for pc in range(min(5, sub["pc_index"].max() + 1 if len(sub) > 0 else 0)):
                    pc_data = sub[sub["pc_index"] == pc]
                    f.write(f"| {tv} | PC{pc} | {pc_data['var_explained'].mean():.4f} | "
                            f"**{pc_data['ucm_alignment'].mean():.4f}** |\n")

        # 4. Phase 1 vs Phase 3 comparison
        f.write("\n## 4. Phase 1 vs Phase 3 Comparison\n\n")
        f.write("| Aspect | Phase 1 (Error UCM) | Phase 3 (Variability UCM) |\n")
        f.write("|---|---|---|\n")
        for tv in tv_names:
            ratios = df[f"{tv}_ratio"].dropna()
            f.write(f"| {tv} median ratio | (see Phase 1 report) | {ratios.median():.4f} |\n")

        # Figures
        f.write("\n## 5. Figures\n\n")
        f.write("- `phase3_ucm_ratio_dist.png` — Per-timestep UCM ratio distributions\n")
        f.write("- `phase3_ratio_vs_action_norm.png` — UCM ratio vs expert action magnitude\n")
        f.write("- `phase3_expert_direction.png` — Variability alignment with expert direction\n")
        f.write("- `phase3_pca_ucm_alignment.png` — PCA components UCM alignment\n")
        f.write("- `phase3_total_variance.png` — Total variance distribution\n")

        f.write("\n---\n*Generated by ucm_analysis Phase 3*\n")

    print(f"Report saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Multi-sample UCM analysis")
    parser.add_argument("--data_dir", default="ucm_analysis/results/multi_samples")
    parser.add_argument("--output_dir", default="ucm_analysis/results/phase3")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Phase 3: Multi-Sample UCM Analysis")
    print("=" * 60)

    # Load data
    print(f"\nLoading multi-sample data from {args.data_dir}...")
    episodes = load_multi_sample_data(args.data_dir)
    total_timesteps = sum(ep["samples"].shape[0] for ep in episodes)
    K = episodes[0]["samples"].shape[1] if episodes else 0
    print(f"Loaded {len(episodes)} episodes, {total_timesteps} timesteps, K={K}")

    # Per-timestep UCM analysis
    print("\nRunning per-timestep UCM decomposition...")
    df = analyze_per_timestep_ucm(episodes, include_gripper=False)
    print(f"  {len(df)} valid timestep results")

    # Print summary
    tv_names = [c.replace("_ratio", "") for c in df.columns if c.endswith("_ratio")]
    print("\n  Summary:")
    for tv in tv_names:
        ratios = df[f"{tv}_ratio"].dropna()
        print(f"    {tv}: median={ratios.median():.4f}, mean={ratios.mean():.4f}, "
              f"pct>1={100*(ratios>1).mean():.1f}%")

    # Expert direction analysis
    print("\nRunning expert direction analysis...")
    expert_dir_df = analyze_expert_direction(episodes)
    if len(expert_dir_df) > 0:
        ratios = expert_dir_df["ratio_along_ortho"].dropna()
        print(f"  Var(along)/Var(ortho): median={ratios.median():.4f}, mean={ratios.mean():.4f}")

    # PCA-UCM alignment
    print("\nRunning PCA-UCM alignment analysis...")
    pca_df = analyze_pca_ucm_alignment(episodes)
    if len(pca_df) > 0:
        for tv in tv_names:
            sub = pca_df[pca_df["tv_name"] == tv]
            pc0 = sub[sub["pc_index"] == 0]
            print(f"  {tv} PC0: mean UCM alignment={pc0['ucm_alignment'].mean():.4f}")

    # Plots
    print("\nGenerating plots...")
    plot_phase3_results(df, expert_dir_df, pca_df, args.output_dir)

    # Report
    print("\nGenerating report...")
    generate_phase3_report(df, expert_dir_df, pca_df, args.output_dir)

    # Save raw data
    df.to_csv(os.path.join(args.output_dir, "per_timestep_ucm.csv"), index=False)
    expert_dir_df.to_csv(os.path.join(args.output_dir, "expert_direction.csv"), index=False)
    print(f"\nRaw data saved to {args.output_dir}/")

    print("\n" + "=" * 60)
    print("Phase 3 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
