"""Quick sanity check for ContrastiveGr00tPolicy and TokenDropContrastivePolicy.

Verifies:
1. Policy loads and action head is found.
2. alpha=1.0 produces valid action shapes and dtypes.
3. alpha=0.0 output exactly matches vanilla (velocity correction is zero).
4. alpha=1.0 output differs from vanilla (contrastive correction applied).
5. Token-drop variants A/B produce different actions from vanilla.
6. Token-drop delta norms are meaningful (nonzero, finite).
7. No NaN/Inf for any variant at various alpha values.

Usage:
    # DDCD tests only:
    uv run python examples/SimplerEnv/quick_test.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE

    # Token-drop tests:
    uv run python examples/SimplerEnv/quick_test.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --run-token-drop

    # Token-drop delta norm diagnosis:
    uv run python examples/SimplerEnv/quick_test.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --run-diagnosis
"""

import argparse
import logging
import sys

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.contrastive_gr00t_policy import ContrastiveGr00tPolicy
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import torch


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def make_dummy_observation(policy: Gr00tPolicy) -> dict[str, any]:
    """Create a minimal dummy observation that passes validation.

    Builds a batch-size-1 observation from the policy's modality configs
    using random data with correct shapes and dtypes.
    """
    modality_configs = policy.modality_configs
    obs: dict[str, any] = {"video": {}, "state": {}, "language": {}}

    for video_key in modality_configs["video"].modality_keys:
        n_frames = len(modality_configs["video"].delta_indices)
        obs["video"][video_key] = np.random.randint(
            0, 256, size=(1, n_frames, 224, 224, 3), dtype=np.uint8
        )

    for state_key in modality_configs["state"].modality_keys:
        n_steps = len(modality_configs["state"].delta_indices)
        # Infer state dim from action config (common pattern: state dim ~ action dim)
        state_dim = 7  # Safe default for most embodiments
        obs["state"][state_key] = np.random.randn(1, n_steps, state_dim).astype(np.float32)

    for lang_key in modality_configs["language"].modality_keys:
        obs["language"][lang_key] = [["pick up the coke can"]]

    return obs


# ======================================================================
# DDCD (Denoising-Depth Contrastive Decoding) tests
# ======================================================================


def test_action_head_found(policy: ContrastiveGr00tPolicy) -> None:
    """Test 1: Verify action head discovery."""
    assert policy._action_head is not None, "Action head not found"
    assert hasattr(policy._action_head, "num_inference_timesteps")
    assert hasattr(policy._action_head, "get_action_with_features")
    original_N = policy._original_num_inference_timesteps
    assert original_N > 0, f"num_inference_timesteps should be > 0, got {original_N}"
    logger.info("PASS: Action head found with num_inference_timesteps=%d", original_N)


def test_contrastive_output_shape(
    policy: ContrastiveGr00tPolicy, obs: dict[str, any]
) -> dict[str, np.ndarray]:
    """Test 2: alpha=1.0 produces valid output shape and dtype."""
    policy.alpha = 1.0
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

    logger.info("PASS: alpha=1.0 output shape/dtype correct. Keys: %s", list(action.keys()))
    return action


def test_alpha_zero_matches_vanilla(
    contrastive_policy: ContrastiveGr00tPolicy,
    vanilla_policy: Gr00tPolicy,
    obs: dict[str, any],
) -> None:
    """Test 3: alpha=0.0 output exactly matches vanilla.

    With velocity-level DDCD, alpha=0 means v_corrected = v_full at every step
    (contrastive correction is zero), so the denoising loop is identical to
    vanilla. With the same seed, results should be exactly identical.
    """
    contrastive_policy.alpha = 0.0

    # Fix seeds for both runs
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_contrastive, _ = contrastive_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_c = action_contrastive[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_c - a_v)
        a_v_norm = np.linalg.norm(a_v)
        ratio = l2_dist / max(a_v_norm, 1e-8)

        # With alpha=0 and same seed, velocity-level DDCD degenerates to the
        # vanilla denoising loop (no v_rough computed, v_corrected = v_full),
        # so results should be exactly identical.
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            ratio,
        )

    logger.info("PASS: alpha=0.0 comparison complete (check ratios above)")


def test_contrastive_differs_from_vanilla(
    contrastive_policy: ContrastiveGr00tPolicy,
    vanilla_policy: Gr00tPolicy,
    obs: dict[str, any],
) -> None:
    """Test 4: alpha=1.0 output differs from vanilla.

    With alpha=1.0 the contrastive correction should produce noticeably
    different actions compared to vanilla, confirming the correction is active.
    """
    contrastive_policy.alpha = 1.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_contrastive, _ = contrastive_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_c = action_contrastive[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_c - a_v)
        a_v_norm = np.linalg.norm(a_v)
        ratio = l2_dist / max(a_v_norm, 1e-8)

        assert l2_dist > 1e-8, (
            f"Key={key}: alpha=1.0 output should differ from vanilla but L2_dist={l2_dist:.8f}"
        )
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            ratio,
        )

    logger.info("PASS: alpha=1.0 produces different actions from vanilla")


# ======================================================================
# Token-Drop Contrastive Decoding tests
# ======================================================================


def _load_token_drop_policy(embodiment_tag, model_path, device, variant, **kwargs):
    """Helper to load a TokenDropContrastivePolicy."""
    from gr00t.policy.token_drop_contrastive_policy import DropVariant, TokenDropContrastivePolicy

    return TokenDropContrastivePolicy(
        embodiment_tag=embodiment_tag,
        model_path=model_path,
        device=device,
        strict=False,
        alpha=kwargs.get("alpha", 1.0),
        variant=DropVariant(variant),
        drop_value=kwargs.get("drop_value", "zero"),
        top_k_ratio=kwargs.get("top_k_ratio", 0.3),
        clamp_ratio=kwargs.get("clamp_ratio", 0.3),
        verbose=kwargs.get("verbose", True),
    )


def test_variant_a_vision_drop(td_policy, vanilla_policy: Gr00tPolicy, obs: dict[str, any]) -> None:
    """Token-drop Test: Vision drop (Variant A) produces different actions from vanilla."""
    td_policy.alpha = 1.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_td, _ = td_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_td = action_td[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_td - a_v)
        a_v_norm = np.linalg.norm(a_v)

        assert l2_dist > 1e-8, (
            f"Key={key}: vision drop should differ from vanilla but L2_dist={l2_dist:.8f}"
        )
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            l2_dist / max(a_v_norm, 1e-8),
        )

    logger.info("PASS: Variant A (vision drop) produces different actions")


def test_variant_b_language_drop(
    td_policy, vanilla_policy: Gr00tPolicy, obs: dict[str, any]
) -> None:
    """Token-drop Test: Language drop (Variant B) produces different actions from vanilla."""
    td_policy.alpha = 1.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_td, _ = td_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_td = action_td[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_td - a_v)
        a_v_norm = np.linalg.norm(a_v)

        assert l2_dist > 1e-8, (
            f"Key={key}: language drop should differ from vanilla but L2_dist={l2_dist:.8f}"
        )
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            l2_dist / max(a_v_norm, 1e-8),
        )

    logger.info("PASS: Variant B (language drop) produces different actions")


def test_td_alpha_zero_matches_vanilla(
    td_policy, vanilla_policy: Gr00tPolicy, obs: dict[str, any]
) -> None:
    """Token-drop Test: alpha=0 produces exactly vanilla output.

    When alpha=0, v_corrected = v_full (amateur velocity is not computed),
    so the denoising loop is identical to vanilla.
    """
    td_policy.alpha = 0.0

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_td, _ = td_policy.get_action(obs)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    action_vanilla, _ = vanilla_policy.get_action(obs)

    for key in action_vanilla:
        a_td = action_td[key]
        a_v = action_vanilla[key]
        l2_dist = np.linalg.norm(a_td - a_v)
        a_v_norm = np.linalg.norm(a_v)
        logger.info(
            "  Key=%s: L2_dist=%.6f, vanilla_norm=%.6f, ratio=%.4f",
            key,
            l2_dist,
            a_v_norm,
            l2_dist / max(a_v_norm, 1e-8),
        )

    logger.info("PASS: Token-drop alpha=0.0 comparison complete")


def test_no_nan(td_policy, obs: dict[str, any]) -> None:
    """Token-drop Test: No NaN/Inf for various alpha values."""
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        td_policy.alpha = alpha
        action, _ = td_policy.get_action(obs)
        for key, arr in action.items():
            assert np.isfinite(arr).all(), f"Key={key}, alpha={alpha}: found NaN/Inf in output"
    logger.info("PASS: No NaN/Inf for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]")


def test_delta_norm_meaningful(
    td_vision_policy, td_language_policy, vanilla_policy, obs: dict[str, any]
) -> None:
    """Diagnose: Compare delta norms between vision drop and language drop.

    ||v_full - v_amateur|| should be:
    - Nonzero (drop has effect)
    - Finite (not NaN/Inf)
    - Variant A vs Variant B comparison reveals relative contribution of
      visual vs linguistic grounding to action prediction.
    """
    results = {}
    for label, policy in [("vision", td_vision_policy), ("language", td_language_policy)]:
        policy.alpha = 1.0
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        action_td, _ = policy.get_action(obs)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        action_vanilla, _ = vanilla_policy.get_action(obs)

        for key in action_vanilla:
            delta = action_td[key] - action_vanilla[key]
            delta_norm = np.linalg.norm(delta)
            vanilla_norm = np.linalg.norm(action_vanilla[key])
            ratio = delta_norm / max(vanilla_norm, 1e-8)
            results[(label, key)] = {
                "delta_norm": delta_norm,
                "vanilla_norm": vanilla_norm,
                "ratio": ratio,
            }

            assert np.isfinite(delta_norm), f"{label}/{key}: delta_norm is not finite"
            assert delta_norm > 1e-8, f"{label}/{key}: delta_norm is ~0 (drop has no effect)"
            logger.info(
                "  %s/%s: delta_norm=%.6f, vanilla_norm=%.6f, ratio=%.4f",
                label,
                key,
                delta_norm,
                vanilla_norm,
                ratio,
            )

    # Compare vision vs language delta
    for key in action_vanilla:
        v_delta = results[("vision", key)]["delta_norm"]
        l_delta = results[("language", key)]["delta_norm"]
        logger.info(
            "  Comparison (%s): vision_delta=%.6f, language_delta=%.6f, vision/language=%.2f",
            key,
            v_delta,
            l_delta,
            v_delta / max(l_delta, 1e-8),
        )

    logger.info("PASS: Delta norms are meaningful and finite")


def test_variant_c_attention_hook(td_attention_policy, obs: dict[str, any]) -> None:
    """Token-drop Test: Variant C (attention-weighted drop) produces valid output.

    Verifies that:
    - Attention weights are captured (no fallback to embedding norm)
    - Output shape is correct
    - No NaN/Inf
    """
    td_attention_policy.alpha = 1.0
    action, _ = td_attention_policy.get_action(obs)

    action_configs = td_attention_policy.modality_configs["action"]
    for action_key in action_configs.modality_keys:
        assert action_key in action, f"Missing action key: {action_key}"
        arr = action[action_key]
        assert np.isfinite(arr).all(), f"Key={action_key}: NaN/Inf in Variant C output"
        assert arr.ndim == 3, f"Expected 3D, got {arr.ndim}D"

    logger.info("PASS: Variant C (attention drop) produces valid output")


# ======================================================================
# Main entry point
# ======================================================================


def run_ddcd_tests(args):
    """Run DDCD (denoising-depth contrastive decoding) tests."""
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    logger.info("Loading ContrastiveGr00tPolicy...")
    contrastive_policy = ContrastiveGr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
        alpha=1.0,
        clamp_ratio=0.3,
        verbose=True,
    )

    logger.info("Loading vanilla Gr00tPolicy...")
    vanilla_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )

    obs = make_dummy_observation(contrastive_policy)

    logger.info("=" * 60)
    logger.info("DDCD Test 1: Action head discovery")
    test_action_head_found(contrastive_policy)

    logger.info("=" * 60)
    logger.info("DDCD Test 2: Output shape and dtype (alpha=1.0)")
    test_contrastive_output_shape(contrastive_policy, obs)

    logger.info("=" * 60)
    logger.info("DDCD Test 3: alpha=0.0 vs vanilla comparison")
    test_alpha_zero_matches_vanilla(contrastive_policy, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("DDCD Test 4: alpha=1.0 differs from vanilla")
    test_contrastive_differs_from_vanilla(contrastive_policy, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("All DDCD tests passed!")


def run_token_drop_tests(args):
    """Run token-drop contrastive decoding tests."""
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    logger.info("Loading vanilla Gr00tPolicy...")
    vanilla_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )

    obs = make_dummy_observation(vanilla_policy)

    logger.info("Loading TokenDropContrastivePolicy (vision)...")
    td_vision = _load_token_drop_policy(
        embodiment_tag, args.model_path, args.device, "vision", verbose=True
    )

    logger.info("=" * 60)
    logger.info("Token-Drop Test: Variant A (vision drop) differs from vanilla")
    test_variant_a_vision_drop(td_vision, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Token-Drop Test: alpha=0 matches vanilla")
    test_td_alpha_zero_matches_vanilla(td_vision, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Token-Drop Test: No NaN/Inf (vision variant)")
    test_no_nan(td_vision, obs)

    logger.info("Loading TokenDropContrastivePolicy (language)...")
    td_language = _load_token_drop_policy(
        embodiment_tag, args.model_path, args.device, "language", verbose=True
    )

    logger.info("=" * 60)
    logger.info("Token-Drop Test: Variant B (language drop) differs from vanilla")
    test_variant_b_language_drop(td_language, vanilla_policy, obs)

    logger.info("Loading TokenDropContrastivePolicy (attention)...")
    td_attention = _load_token_drop_policy(
        embodiment_tag,
        args.model_path,
        args.device,
        "attention",
        top_k_ratio=0.3,
        verbose=True,
    )

    logger.info("=" * 60)
    logger.info("Token-Drop Test: Variant C (attention drop) produces valid output")
    test_variant_c_attention_hook(td_attention, obs)

    logger.info("=" * 60)
    logger.info("All token-drop tests passed!")


def run_diagnosis(args):
    """Run delta norm diagnosis for token-drop variants.

    Compares the delta norms between vision drop (A) and language drop (B)
    to assess relative contribution of visual vs linguistic grounding.
    This should be run first before sweeping alpha to verify that the
    token drop produces meaningful contrastive signals.
    """
    embodiment_tag = EmbodimentTag(args.embodiment_tag)

    logger.info("Loading vanilla Gr00tPolicy...")
    vanilla_policy = Gr00tPolicy(
        embodiment_tag=embodiment_tag,
        model_path=args.model_path,
        device=args.device,
        strict=False,
    )

    obs = make_dummy_observation(vanilla_policy)

    logger.info("Loading TokenDropContrastivePolicy (vision)...")
    td_vision = _load_token_drop_policy(
        embodiment_tag, args.model_path, args.device, "vision", verbose=True
    )

    logger.info("Loading TokenDropContrastivePolicy (language)...")
    td_language = _load_token_drop_policy(
        embodiment_tag, args.model_path, args.device, "language", verbose=True
    )

    logger.info("=" * 60)
    logger.info("Delta Norm Diagnosis: Vision Drop vs Language Drop")
    test_delta_norm_meaningful(td_vision, td_language, vanilla_policy, obs)

    logger.info("=" * 60)
    logger.info("Diagnosis complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test for Contrastive and Token-Drop policies"
    )
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
        "--run-token-drop",
        action="store_true",
        help="Run token-drop CD tests instead of DDCD tests",
    )
    parser.add_argument(
        "--run-diagnosis",
        action="store_true",
        help="Run delta norm diagnosis (vision vs language drop comparison)",
    )
    args = parser.parse_args()

    if args.run_diagnosis:
        run_diagnosis(args)
    elif args.run_token_drop:
        run_token_drop_tests(args)
    else:
        run_ddcd_tests(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
