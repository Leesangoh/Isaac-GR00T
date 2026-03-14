"""
Phase 1: UCM Decomposition of VLA-Expert Error
================================================
Analyzes whether VLA prediction errors are structured (concentrated in
task-irrelevant dimensions) or uniformly distributed.

Uses existing extracted data: action_expert (T,7) and action_vla_chunks (T,8,7)
from /mnt/md1/solee/features/vla_actions/
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from glob import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ucm_analysis.config import VLA_ACTIONS_DIR, N_STEPS, ACTION_DIM, ACTION_LABELS
from ucm_analysis.task_variables import TASK_VARIABLES, TASK_VARIABLES_NO_GRIPPER
from ucm_analysis.ucm_decomposition import compute_ucm_decomposition, compute_ucm_per_episode
from ucm_analysis.phase_segmentation import segment_episode_phases, PHASE_NAMES
from ucm_analysis.statistical_tests import bootstrap_ucm_ratio, permutation_test_ucm
from ucm_analysis.visualization import (
    plot_ucm_ratio_bar,
    plot_ucm_ratio_distribution,
    plot_phase_ucm,
    plot_chunk_step_ucm,
    plot_error_projection_scatter,
    plot_variance_eigenspectrum,
)


def load_episodes(vla_dir, max_episodes=None):
    """Load all episodes from VLA actions directory."""
    files = sorted(glob(os.path.join(vla_dir, "episode_*.pt")))
    if max_episodes:
        files = files[:max_episodes]

    episodes = []
    for f in files:
        data = torch.load(f, map_location='cpu', weights_only=False)
        episodes.append({
            "episode_id": data["episode_id"],
            "action_expert": data["action_expert"].numpy(),      # (T, 7)
            "action_vla_chunks": data["action_vla_chunks"].numpy(),  # (T, 8, 7)
        })
    return episodes


def build_error_vectors(episodes, include_gripper=False):
    """
    Build error vectors from episodes.

    For each timestep, construct the 8-step expert action chunk and compute
    error = VLA_chunk - expert_chunk.

    Returns:
        all_errors: (N_total, D) array of flattened error vectors
        episode_errors: list of (T_i, D) arrays per episode
        episode_expert_actions: list of (T_i, 7) expert step-0 actions
    """
    all_errors = []
    episode_errors = []
    episode_expert_actions = []

    for ep in episodes:
        expert = ep["action_expert"]       # (T, 7)
        vla_chunks = ep["action_vla_chunks"]  # (T, 8, 7)
        T = len(expert)

        # Build expert action chunks: for timestep t, expert chunk is expert[t:t+8]
        ep_errors = []
        for t in range(T):
            # Expert chunk: expert actions from t to t+7 (pad with last if needed)
            expert_chunk = np.zeros((N_STEPS, ACTION_DIM), dtype=np.float32)
            for s in range(N_STEPS):
                idx = min(t + s, T - 1)
                expert_chunk[s] = expert[idx]

            vla_chunk = vla_chunks[t]  # (8, 7)

            if include_gripper:
                error = (vla_chunk - expert_chunk).flatten()  # 56D
            else:
                error = (vla_chunk[:, :6] - expert_chunk[:, :6]).flatten()  # 48D

            ep_errors.append(error)

        ep_errors = np.array(ep_errors)  # (T, D)
        all_errors.append(ep_errors)
        episode_errors.append(ep_errors)
        episode_expert_actions.append(expert)

    all_errors = np.concatenate(all_errors, axis=0)
    return all_errors, episode_errors, episode_expert_actions


def build_per_step_errors(episodes, include_gripper=False):
    """Build errors per chunk step for step-wise analysis."""
    step_errors = defaultdict(list)

    for ep in episodes:
        expert = ep["action_expert"]
        vla_chunks = ep["action_vla_chunks"]
        T = len(expert)

        for t in range(T):
            for s in range(N_STEPS):
                idx = min(t + s, T - 1)
                if include_gripper:
                    error = vla_chunks[t, s] - expert[idx]  # 7D
                else:
                    error = vla_chunks[t, s, :6] - expert[idx, :6]  # 6D
                step_errors[s].append(error)

    return {s: np.array(v) for s, v in step_errors.items()}


def run_global_ucm_analysis(all_errors, include_gripper=False):
    """Run UCM decomposition for all task variables."""
    tv_dict = TASK_VARIABLES if include_gripper else TASK_VARIABLES_NO_GRIPPER
    results = {}

    for tv_name, build_J in tv_dict.items():
        if include_gripper:
            J = build_J(n_steps=N_STEPS, action_dim=ACTION_DIM)
        else:
            J = build_J(n_steps=N_STEPS, action_dim=6)
        result = compute_ucm_decomposition(all_errors, J)
        results[tv_name] = result
        print(f"  {tv_name}: V_UCM/dof={result['V_ucm']:.6f}, V_ORT/dof={result['V_ort']:.6f}, "
              f"ratio={result['ratio']:.4f}, UCM_dim={result['d_ucm']}, ORT_dim={result['d_ort']}")

    return results


def run_phase_analysis(episodes, episode_errors, episode_expert_actions, include_gripper=False):
    """Run UCM analysis per task phase."""
    tv_dict = TASK_VARIABLES if include_gripper else TASK_VARIABLES_NO_GRIPPER
    phase_errors = defaultdict(list)

    for i, ep in enumerate(episodes):
        expert = episode_expert_actions[i]
        labels = segment_episode_phases(expert)
        errors = episode_errors[i]

        for t in range(len(errors)):
            phase_errors[labels[t]].append(errors[t])

    phase_results = {}
    for phase_id in sorted(phase_errors.keys()):
        errs = np.array(phase_errors[phase_id])
        phase_name = PHASE_NAMES.get(phase_id, str(phase_id))
        print(f"\n  Phase: {phase_name} (N={len(errs)})")

        phase_results[phase_id] = {}
        for tv_name, build_J in tv_dict.items():
            if include_gripper:
                J = build_J(n_steps=N_STEPS, action_dim=ACTION_DIM)
            else:
                J = build_J(n_steps=N_STEPS, action_dim=6)

            if len(errs) < 10:
                print(f"    {tv_name}: too few samples")
                phase_results[phase_id][tv_name] = {"V_ucm": 0, "V_ort": 0, "ratio": float('nan')}
                continue

            result = compute_ucm_decomposition(errs, J)
            phase_results[phase_id][tv_name] = result
            print(f"    {tv_name}: ratio={result['ratio']:.4f}")

    return phase_results


def run_step_analysis(episodes, include_gripper=False):
    """UCM analysis per chunk step (using single-step 7D or 6D errors)."""
    step_errors = build_per_step_errors(episodes, include_gripper=include_gripper)
    dim = ACTION_DIM if include_gripper else 6

    # For single-step analysis, task variable is position [x,y,z]
    # J is (3, dim): identity for first 3 dims
    J_pos = np.zeros((3, dim))
    J_pos[0, 0] = 1.0
    J_pos[1, 1] = 1.0
    J_pos[2, 2] = 1.0

    step_results = {}
    for s in sorted(step_errors.keys()):
        errs = step_errors[s]
        result = compute_ucm_decomposition(errs, J_pos)
        step_results[s] = {"single_step_position": result}
        print(f"  Step {s}: ratio={result['ratio']:.4f}, V_UCM={result['V_ucm']:.6f}, V_ORT={result['V_ort']:.6f}")

    return step_results


def run_statistical_tests(episode_errors, include_gripper=False):
    """Run bootstrap and permutation tests."""
    tv_dict = TASK_VARIABLES if include_gripper else TASK_VARIABLES_NO_GRIPPER
    stats_results = {}

    for tv_name, build_J in tv_dict.items():
        if include_gripper:
            J = build_J(n_steps=N_STEPS, action_dim=ACTION_DIM)
        else:
            J = build_J(n_steps=N_STEPS, action_dim=6)

        # Per-episode ratios
        ep_ratios = compute_ucm_per_episode(episode_errors, J)

        # Bootstrap
        boot = bootstrap_ucm_ratio(ep_ratios)
        print(f"  {tv_name}: mean={boot['mean']:.4f}, median={boot['median']:.4f}, "
              f"95% CI=[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}], p={boot['p_value']:.2e}")

        stats_results[tv_name] = {
            "bootstrap": boot,
            "episode_ratios": ep_ratios,
        }

    return stats_results


def run_permutation_tests(all_errors, include_gripper=False, n_permutations=2000):
    """Run permutation tests (slower, on subsampled data)."""
    tv_dict = TASK_VARIABLES if include_gripper else TASK_VARIABLES_NO_GRIPPER

    # Subsample for speed
    n = len(all_errors)
    if n > 10000:
        idx = np.random.choice(n, 10000, replace=False)
        errors_sub = all_errors[idx]
    else:
        errors_sub = all_errors

    perm_results = {}
    for tv_name, build_J in tv_dict.items():
        if include_gripper:
            J = build_J(n_steps=N_STEPS, action_dim=ACTION_DIM)
        else:
            J = build_J(n_steps=N_STEPS, action_dim=6)

        result = permutation_test_ucm(errors_sub, J, n_permutations=n_permutations)
        print(f"  {tv_name}: observed={result['observed_ratio']:.4f}, "
              f"perm_mean={result['perm_mean']:.4f}, p={result['p_value']:.4f}")
        perm_results[tv_name] = result

    return perm_results


def per_dimension_analysis(episodes):
    """Compute per-dimension error statistics."""
    all_expert = []
    all_vla_step0 = []

    for ep in episodes:
        all_expert.append(ep["action_expert"])
        all_vla_step0.append(ep["action_vla_chunks"][:, 0, :])  # Step 0 only

    expert = np.concatenate(all_expert, axis=0)
    vla = np.concatenate(all_vla_step0, axis=0)
    error = vla - expert

    print("\n  Per-dimension error stats (step 0):")
    print(f"  {'Dim':>8} {'MAE':>10} {'RMSE':>10} {'Expert_std':>10} {'VLA_std':>10} {'R²':>8}")
    for d in range(ACTION_DIM):
        mae = np.mean(np.abs(error[:, d]))
        rmse = np.sqrt(np.mean(error[:, d] ** 2))
        expert_std = np.std(expert[:, d])
        vla_std = np.std(vla[:, d])
        ss_res = np.sum(error[:, d] ** 2)
        ss_tot = np.sum((expert[:, d] - np.mean(expert[:, d])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  {ACTION_LABELS[d]:>8} {mae:10.6f} {rmse:10.6f} {expert_std:10.6f} {vla_std:10.6f} {r2:8.4f}")

    return error, expert, vla


def generate_report(results, phase_results, step_results, stats_results,
                    perm_results, dim_stats, output_dir):
    """Generate markdown report."""
    report_path = os.path.join(output_dir, "phase1_results.md")
    with open(report_path, 'w') as f:
        f.write("# Phase 1: UCM Decomposition of VLA-Expert Error\n\n")
        f.write("## Overview\n\n")
        f.write("This analysis decomposes VLA (GR00T N1.6) prediction errors against expert actions\n")
        f.write("from BridgeData V2 using the Uncontrolled Manifold (UCM) framework.\n\n")
        f.write("**Key question**: Is VLA error concentrated in task-irrelevant dimensions (UCM hypothesis)?\n\n")
        f.write(f"- UCM ratio > 1: Error concentrated in task-irrelevant space (supports hypothesis)\n")
        f.write(f"- UCM ratio = 1: Error uniformly distributed (null hypothesis)\n")
        f.write(f"- UCM ratio < 1: Error concentrated in task-relevant space (opposes hypothesis)\n\n")

        # Global results
        f.write("## 1. Global UCM Decomposition (gripper excluded, 48D)\n\n")
        f.write("| Task Variable | V_UCM/dof | V_ORT/dof | Ratio | UCM dim | ORT dim |\n")
        f.write("|---|---|---|---|---|---|\n")
        for tv_name, res in results.items():
            f.write(f"| {tv_name} | {res['V_ucm']:.6f} | {res['V_ort']:.6f} | "
                    f"**{res['ratio']:.4f}** | {res['d_ucm']} | {res['d_ort']} |\n")

        # Interpretation
        f.write("\n### Interpretation\n\n")
        for tv_name, res in results.items():
            ratio = res['ratio']
            if ratio > 1.5:
                interp = "Strong UCM effect — error is concentrated in task-irrelevant dimensions"
            elif ratio > 1.1:
                interp = "Moderate UCM effect"
            elif ratio > 0.9:
                interp = "Near-uniform variance distribution (no clear UCM structure)"
            else:
                interp = "Inverse UCM — error is concentrated in task-relevant dimensions"
            f.write(f"- **{tv_name}**: ratio={ratio:.4f} → {interp}\n")

        # Statistical tests
        f.write("\n## 2. Statistical Tests\n\n")
        f.write("### Bootstrap (per-episode UCM ratios)\n\n")
        f.write("| Task Variable | Mean | Median | 95% CI | p-value (H0: ratio=1) | N episodes |\n")
        f.write("|---|---|---|---|---|---|\n")
        for tv_name, sr in stats_results.items():
            b = sr["bootstrap"]
            f.write(f"| {tv_name} | {b['mean']:.4f} | {b['median']:.4f} | "
                    f"[{b['ci_low']:.4f}, {b['ci_high']:.4f}] | {b['p_value']:.2e} | {b['n']} |\n")

        f.write("\n### Permutation Test (dimension shuffling)\n\n")
        f.write("| Task Variable | Observed Ratio | Permutation Mean | p-value |\n")
        f.write("|---|---|---|---|\n")
        for tv_name, pr in perm_results.items():
            f.write(f"| {tv_name} | {pr['observed_ratio']:.4f} | {pr['perm_mean']:.4f} | {pr['p_value']:.4f} |\n")

        # Phase analysis
        f.write("\n## 3. Task Phase Analysis\n\n")
        for phase_id in sorted(phase_results.keys()):
            phase_name = PHASE_NAMES.get(phase_id, str(phase_id))
            f.write(f"\n### Phase: {phase_name}\n\n")
            f.write("| Task Variable | V_UCM/dof | V_ORT/dof | Ratio |\n")
            f.write("|---|---|---|---|\n")
            for tv_name, res in phase_results[phase_id].items():
                if isinstance(res.get('ratio'), float) and np.isnan(res['ratio']):
                    f.write(f"| {tv_name} | N/A | N/A | N/A |\n")
                else:
                    f.write(f"| {tv_name} | {res['V_ucm']:.6f} | {res['V_ort']:.6f} | **{res['ratio']:.4f}** |\n")

        # Step analysis
        f.write("\n## 4. Chunk Step Analysis (single-step, position task variable)\n\n")
        f.write("| Step | V_UCM/dof | V_ORT/dof | Ratio |\n")
        f.write("|---|---|---|---|\n")
        for s in sorted(step_results.keys()):
            res = step_results[s]["single_step_position"]
            f.write(f"| {s} | {res['V_ucm']:.6f} | {res['V_ort']:.6f} | **{res['ratio']:.4f}** |\n")

        # Figures
        f.write("\n## 5. Figures\n\n")
        f.write("- `ucm_ratio_bar.png` — V_UCM vs V_ORT per task variable\n")
        f.write("- `ucm_ratio_distribution.png` — Per-episode ratio distributions\n")
        f.write("- `phase_ucm.png` — UCM ratio by task phase\n")
        f.write("- `chunk_step_ucm.png` — UCM ratio across chunk steps\n")
        f.write("- `error_projection_scatter.png` — UCM vs ORT error projections\n")
        f.write("- `eigenspectrum.png` — Error covariance eigenspectrum\n")

        f.write("\n---\n")
        f.write("*Generated by ucm_analysis Phase 1*\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    parser = argparse.ArgumentParser(description="Phase 1: UCM decomposition of VLA-Expert errors")
    parser.add_argument("--vla_dir", default=VLA_ACTIONS_DIR)
    parser.add_argument("--output_dir", default="ucm_analysis/results/phase1")
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit episodes for testing")
    parser.add_argument("--n_permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # === Load Data ===
    print("=" * 60)
    print("Phase 1: UCM Decomposition of VLA-Expert Error")
    print("=" * 60)

    print(f"\nLoading episodes from {args.vla_dir}...")
    episodes = load_episodes(args.vla_dir, max_episodes=args.max_episodes)
    total_timesteps = sum(len(ep["action_expert"]) for ep in episodes)
    print(f"Loaded {len(episodes)} episodes, {total_timesteps} total timesteps")

    # === Build Error Vectors (excluding gripper) ===
    print("\nBuilding error vectors (excluding gripper, 48D)...")
    all_errors, episode_errors, episode_expert_actions = build_error_vectors(episodes, include_gripper=False)
    print(f"Error matrix shape: {all_errors.shape}")

    # === Per-Dimension Stats ===
    print("\n" + "=" * 60)
    print("Per-Dimension Error Statistics")
    print("=" * 60)
    dim_stats = per_dimension_analysis(episodes)

    # === Global UCM Analysis ===
    print("\n" + "=" * 60)
    print("Global UCM Decomposition")
    print("=" * 60)
    results = run_global_ucm_analysis(all_errors, include_gripper=False)

    # === Phase Analysis ===
    print("\n" + "=" * 60)
    print("Task Phase UCM Analysis")
    print("=" * 60)
    phase_results = run_phase_analysis(episodes, episode_errors, episode_expert_actions, include_gripper=False)

    # === Chunk Step Analysis ===
    print("\n" + "=" * 60)
    print("Chunk Step UCM Analysis")
    print("=" * 60)
    step_results = run_step_analysis(episodes, include_gripper=False)

    # === Statistical Tests ===
    print("\n" + "=" * 60)
    print("Statistical Tests (Bootstrap)")
    print("=" * 60)
    stats_results = run_statistical_tests(episode_errors, include_gripper=False)

    print("\n" + "=" * 60)
    print("Statistical Tests (Permutation)")
    print("=" * 60)
    perm_results = run_permutation_tests(all_errors, include_gripper=False, n_permutations=args.n_permutations)

    # === Visualization ===
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    plot_ucm_ratio_bar(results, os.path.join(fig_dir, "ucm_ratio_bar.png"))
    print("  Saved ucm_ratio_bar.png")

    ep_ratios_dict = {tv: sr["episode_ratios"] for tv, sr in stats_results.items()}
    plot_ucm_ratio_distribution(ep_ratios_dict, os.path.join(fig_dir, "ucm_ratio_distribution.png"))
    print("  Saved ucm_ratio_distribution.png")

    # Phase plot - only if we have phase data with valid results
    valid_phases = {p: v for p, v in phase_results.items()
                    if any(not (isinstance(r.get('ratio'), float) and np.isnan(r.get('ratio', float('nan'))))
                           for r in v.values())}
    if valid_phases:
        plot_phase_ucm(valid_phases, os.path.join(fig_dir, "phase_ucm.png"))
        print("  Saved phase_ucm.png")

    plot_chunk_step_ucm(step_results, os.path.join(fig_dir, "chunk_step_ucm.png"))
    print("  Saved chunk_step_ucm.png")

    plot_error_projection_scatter(results, os.path.join(fig_dir, "error_projection_scatter.png"))
    print("  Saved error_projection_scatter.png")

    # Eigenspectrum for TV2 (most balanced UCM/ORT split)
    tv_dict = TASK_VARIABLES_NO_GRIPPER
    J_tv2 = tv_dict["TV2_cumulative_pos_orient"](n_steps=N_STEPS, action_dim=6)
    eigenvalues, ucm_alignment = plot_variance_eigenspectrum(
        all_errors, J_tv2, os.path.join(fig_dir, "eigenspectrum.png"))
    print("  Saved eigenspectrum.png")

    # === Generate Report ===
    print("\n" + "=" * 60)
    print("Generating Report")
    print("=" * 60)
    report_path = generate_report(results, phase_results, step_results, stats_results,
                                  perm_results, dim_stats, args.output_dir)

    print("\n" + "=" * 60)
    print("Phase 1 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
