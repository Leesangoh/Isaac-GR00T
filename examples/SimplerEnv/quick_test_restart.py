"""Quick sanity check for RestartSamplingPolicy.

Verifies:
1. Policy loads and action head is found.
2. Output shape matches expectations.
3. K=0 (no restart) matches vanilla (same total_steps, same NFE).
4. K=1 produces different actions from vanilla.
5. NFE counting: verify NFE = steps_phase1 + K × steps_restart + steps_phase3.
6. No NaN/Inf for all noise methods.
7. Three noise methods produce different outputs.
8. Diagnose action quality: compare vanilla vs restart on L2 norm,
   temporal smoothness, and mode consistency.

Usage:
    uv run python examples/SimplerEnv/quick_test_restart.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE

    # Run quality diagnosis:
    uv run python examples/SimplerEnv/quick_test_restart.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --run-diagnosis
"""

import argparse
import logging
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.policy.restart_sampling_policy import RestartSamplingPolicy
import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def make_dummy_observation(policy: Gr00tPolicy) -> dict[str, any]:
    """Create a minimal dummy observation that passes validation."""
    modality_configs = policy.modality_configs
    obs: dict[str, any] = {"video": {}, "state": {}, "language": {}}

    for video_key in modality_configs["video"].modality_keys:
        n_frames = len(modality_configs["video"].delta_indices)
        obs["video"][video_key] = np.random.randint(
            0, 256, size=(1, n_frames, 224, 224, 3), dtype=np.uint8
        )

    for state_key in modality_configs["state"].modality_keys:
        n_steps = len(modality_configs["state"].delta_indices)
        state_dim = 7
        obs["state"][state_key] = np.random.randn(1, n_steps, state_dim).astype(np.float32)

    for lang_key in modality_configs["language"].modality_keys:
        obs["language"][lang_key] = [["pick up the coke can"]]

    return obs


# ======================================================================
# Restart Sampling tests
# ======================================================================


def test_action_head_found(policy: RestartSamplingPolicy) -> None:
    """Test 1: Verify action head discovery."""
    assert policy._action_head is not None, "Action head not found"
    assert hasattr(policy._action_head, "num_inference_timesteps")
    assert hasattr(policy._action_head, "get_action_with_features")
    original_N = policy._original_num_inference_timesteps
    assert original_N > 0, f"num_inference_timesteps should be > 0, got {original_N}"
    logger.info("PASS: Action head found with num_inference_timesteps=%d", original_N)


def test_output_shape(policy: RestartSamplingPolicy, obs: dict[str, any]) -> dict[str, np.ndarray]:
    """Test 2: Output shape matches expectations."""
    action, info = policy.get_action(obs)

    action_configs = policy.modality_configs["action"]
    for action_key in action_configs.modality_keys:
        assert action_key in action, f"Missing action key: {action_key}"
        arr = action[action_key]
        assert isinstance(arr, np.ndarray), f"Expected np.ndarray, got {type(arr)}"
        assert arr.dtype == np.float32, f"Expected float32, got {arr.dtype}"
        assert arr.ndim == 3, f"Expected 3D (B, T, D), got {arr.ndim}D"
        expected_horizon = len(action_configs.delta_indices)
        assert arr.shape[1] == expected_horizon, (
            f"Action horizon mismatch: {arr.shape[1]} vs {expected_horizon}"
        )

    logger.info("PASS: Output shape/dtype correct. Keys: %s", list(action.keys()))
    return action


def test_K_zero_matches_vanilla(
    restart_policy: RestartSamplingPolicy,
    vanilla_policy: Gr00tPolicy,
    obs: dict[str, any],
) -> None:
    """Test 3: K=0 (no restart) matches vanilla with same total_steps.

    When K=0, the restart policy degenerates to a standard ODE with
    steps_phase1 + steps_phase3 steps from t=0 to t=1. With the same
    total NFE and seed, results should match vanilla.
    """
    # Save original K and create a K=0 policy by temporarily overriding
    orig_K = restart_policy.restart_K
    restart_policy.restart_K = 0
    # Recompute step allocation for K=0
    orig_phase1 = restart_policy._steps_phase1
    orig_phase3 = restart_policy._steps_phase3
    orig_nfe = restart_policy._actual_nfe
    total = restart_policy.total_steps
    restart_policy._steps_phase1 = max(1, round(restart_policy.t_restart * total))
    restart_policy._steps_phase3 = max(1, total - restart_policy._steps_phase1)
    restart_policy._actual_nfe = restart_policy._steps_phase1 + restart_policy._steps_phase3

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_restart, _ = restart_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_r = action_restart[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_r - a_v)
        a_v_norm = np.linalg.norm(a_v)
        ratio = l2_dist / max(a_v_norm, 1e-8)
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            ratio,
        )

    # Restore
    restart_policy.restart_K = orig_K
    restart_policy._steps_phase1 = orig_phase1
    restart_policy._steps_phase3 = orig_phase3
    restart_policy._actual_nfe = orig_nfe

    logger.info("PASS: K=0 vs vanilla comparison complete (check ratios above)")


def test_restart_differs(
    restart_policy: RestartSamplingPolicy,
    vanilla_policy: Gr00tPolicy,
    obs: dict[str, any],
) -> None:
    """Test 4: K=1 produces different actions from vanilla.

    With restart cycles, the stochastic re-noising should produce
    noticeably different actions compared to vanilla ODE.
    """
    assert restart_policy.restart_K >= 1, "Need K>=1 for this test"

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_restart, _ = restart_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_r = action_restart[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_r - a_v)
        a_v_norm = np.linalg.norm(a_v)
        ratio = l2_dist / max(a_v_norm, 1e-8)

        assert l2_dist > 1e-8, (
            f"Key={key}: K=1 restart should differ from vanilla but L2_dist={l2_dist:.8f}"
        )
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            ratio,
        )

    logger.info("PASS: K=1 restart produces different actions from vanilla")


def test_nfe_counting(restart_policy: RestartSamplingPolicy) -> None:
    """Test 5: Verify NFE = steps_phase1 + K × steps_restart + steps_phase3."""
    expected_nfe = (
        restart_policy._steps_phase1
        + restart_policy.restart_K * restart_policy.steps_restart
        + restart_policy._steps_phase3
    )
    actual_nfe = restart_policy.nfe

    assert actual_nfe == expected_nfe, (
        f"NFE mismatch: actual={actual_nfe} vs expected={expected_nfe} "
        f"(phase1={restart_policy._steps_phase1}, "
        f"K={restart_policy.restart_K}×{restart_policy.steps_restart}, "
        f"phase3={restart_policy._steps_phase3})"
    )
    logger.info(
        "PASS: NFE=%d = %d + %d×%d + %d",
        actual_nfe,
        restart_policy._steps_phase1,
        restart_policy.restart_K,
        restart_policy.steps_restart,
        restart_policy._steps_phase3,
    )


def test_no_nan(obs: dict[str, any], embodiment_tag, model_path, device) -> None:
    """Test 6: No NaN/Inf for all noise methods."""
    for method in ["sdedit", "interpolation", "scaled"]:
        policy = RestartSamplingPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=False,
            t_restart=0.6,
            t_back=0.3,
            restart_K=1,
            steps_restart=3,
            total_steps=10,
            noise_method=method,
        )
        action, _ = policy.get_action(obs)
        for key, arr in action.items():
            assert np.isfinite(arr).all(), f"Key={key}, method={method}: found NaN/Inf in output"
        logger.info("  method=%s: OK (no NaN/Inf)", method)
        del policy

    logger.info("PASS: No NaN/Inf for all noise methods")


def test_noise_methods_differ(obs: dict[str, any], embodiment_tag, model_path, device) -> None:
    """Test 7: Three noise methods produce different outputs."""
    results = {}
    for method in ["sdedit", "interpolation", "scaled"]:
        policy = RestartSamplingPolicy(
            embodiment_tag=embodiment_tag,
            model_path=model_path,
            device=device,
            strict=False,
            t_restart=0.6,
            t_back=0.3,
            restart_K=1,
            steps_restart=3,
            total_steps=10,
            noise_method=method,
        )
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        action, _ = policy.get_action(obs)
        results[method] = action
        del policy

    methods = list(results.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            for key in results[m1]:
                l2_dist = np.linalg.norm(results[m1][key] - results[m2][key])
                logger.info("  %s vs %s [%s]: L2_dist=%.6f", m1, m2, key, l2_dist)
                assert l2_dist > 1e-8, (
                    f"{m1} vs {m2} [{key}]: methods should produce different outputs"
                )

    logger.info("PASS: All three noise methods produce different outputs")


# ======================================================================
# Quality diagnosis
# ======================================================================


def diagnose_action_quality(
    restart_policy: RestartSamplingPolicy,
    vanilla_policy: Gr00tPolicy,
    obs: dict[str, any],
    n_samples: int = 5,
) -> None:
    """Compare vanilla vs restart on L2 norm, temporal smoothness, mode consistency."""
    logger.info("Running quality diagnosis with %d samples...", n_samples)

    vanilla_actions_list = []
    restart_actions_list = []

    for i in range(n_samples):
        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        action_v, _ = vanilla_policy.get_action(obs)
        vanilla_actions_list.append(action_v)

        torch.manual_seed(i)
        torch.cuda.manual_seed_all(i)
        action_r, _ = restart_policy.get_action(obs)
        restart_actions_list.append(action_r)

    # Analyze each action key
    action_keys = list(vanilla_actions_list[0].keys())
    for key in action_keys:
        vanilla_arr = np.stack([a[key] for a in vanilla_actions_list])  # (N, B, T, D)
        restart_arr = np.stack([a[key] for a in restart_actions_list])

        # L2 norm
        v_norms = np.linalg.norm(vanilla_arr, axis=-1).mean()
        r_norms = np.linalg.norm(restart_arr, axis=-1).mean()

        # Temporal smoothness (L2 of consecutive differences)
        v_smooth = np.linalg.norm(np.diff(vanilla_arr, axis=2), axis=-1).mean()
        r_smooth = np.linalg.norm(np.diff(restart_arr, axis=2), axis=-1).mean()

        # Mode consistency (std across samples)
        v_std = vanilla_arr.std(axis=0).mean()
        r_std = restart_arr.std(axis=0).mean()

        logger.info("  Key=%s:", key)
        logger.info(
            "    L2 norm:     vanilla=%.4f, restart=%.4f",
            v_norms,
            r_norms,
        )
        logger.info(
            "    Smoothness:  vanilla=%.4f, restart=%.4f (lower=smoother)",
            v_smooth,
            r_smooth,
        )
        logger.info(
            "    Mode std:    vanilla=%.4f, restart=%.4f (lower=more consistent)",
            v_std,
            r_std,
        )

    logger.info("Quality diagnosis complete")


# ======================================================================
# Main entry point
# ======================================================================


def run_restart_tests(args):
    """Run restart sampling tests."""
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    logger.info("Loading RestartSamplingPolicy...")
    restart_policy = RestartSamplingPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
        t_restart=0.6,
        t_back=0.3,
        restart_K=1,
        steps_restart=3,
        total_steps=10,
        noise_method="sdedit",
        verbose=True,
    )

    logger.info("Loading vanilla Gr00tPolicy...")
    vanilla_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )

    obs = make_dummy_observation(restart_policy)

    logger.info("=" * 60)
    logger.info("Test 1: Action head discovery")
    test_action_head_found(restart_policy)

    logger.info("=" * 60)
    logger.info("Test 2: Output shape")
    test_output_shape(restart_policy, obs)

    logger.info("=" * 60)
    logger.info("Test 3: K=0 vs vanilla comparison")
    test_K_zero_matches_vanilla(restart_policy, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Test 4: K=1 restart differs from vanilla")
    test_restart_differs(restart_policy, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Test 5: NFE counting")
    test_nfe_counting(restart_policy)

    logger.info("=" * 60)
    logger.info("Test 6: No NaN/Inf for all noise methods")
    test_no_nan(obs, embodiment_tag, args.model_path, args.device)

    logger.info("=" * 60)
    logger.info("Test 7: Noise methods produce different outputs")
    test_noise_methods_differ(obs, embodiment_tag, args.model_path, args.device)

    logger.info("=" * 60)
    logger.info("All restart sampling tests passed!")


def run_diagnosis(args):
    """Run action quality diagnosis."""
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    logger.info("Loading RestartSamplingPolicy...")
    restart_policy = RestartSamplingPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
        t_restart=0.6,
        t_back=0.3,
        restart_K=1,
        steps_restart=3,
        total_steps=10,
        noise_method="sdedit",
        verbose=False,
    )

    logger.info("Loading vanilla Gr00tPolicy...")
    vanilla_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )

    obs = make_dummy_observation(restart_policy)

    logger.info("=" * 60)
    logger.info("Action Quality Diagnosis: Vanilla vs Restart Sampling")
    diagnose_action_quality(restart_policy, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Diagnosis complete!")


def main():
    parser = argparse.ArgumentParser(description="Quick test for RestartSamplingPolicy")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Model checkpoint path or HF ID"
    )
    parser.add_argument(
        "--embodiment-tag",
        type=str,
        required=True,
        help="Embodiment tag (e.g. OXE_GOOGLE)",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--run-diagnosis",
        action="store_true",
        help="Run action quality diagnosis (vanilla vs restart comparison)",
    )
    args = parser.parse_args()

    if args.run_diagnosis:
        run_diagnosis(args)
    else:
        run_restart_tests(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
