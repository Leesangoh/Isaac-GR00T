"""GR00T inference server with VLM Token-Drop Contrastive Decoding.

Drop-in replacement for gr00t/eval/run_gr00t_server.py that uses
TokenDropContrastivePolicy instead of the vanilla Gr00tPolicy.

Usage:
    uv run python examples/SimplerEnv/run_token_drop_server.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --use-sim-policy-wrapper \
        --variant vision \
        --alpha 1.0

    # Variant B (language drop):
    uv run python examples/SimplerEnv/run_token_drop_server.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --use-sim-policy-wrapper \
        --variant language \
        --alpha 1.0

    # Variant C (attention-weighted drop):
    uv run python examples/SimplerEnv/run_token_drop_server.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --use-sim-policy-wrapper \
        --variant attention \
        --alpha 1.0 \
        --top-k-ratio 0.3
"""

from dataclasses import dataclass
import logging
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import PolicyServer
from gr00t.policy.token_drop_contrastive_policy import DropVariant, TokenDropContrastivePolicy
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class TokenDropServerConfig:
    """Configuration for GR00T Token-Drop CD inference server."""

    # Gr00t policy configs
    model_path: str = ""
    """Path to the model checkpoint directory or HuggingFace model ID."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag for the robot type."""

    device: str = "cuda"
    """Device to run the model on."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input and output validation."""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper for SimplerEnv compatibility."""

    # Token-drop CD configs
    alpha: float = 1.0
    """Contrastive amplification factor. 0.0 = vanilla, 1.0 = standard CFG."""

    variant: str = "vision"
    """Drop variant: 'vision' (A), 'language' (B), or 'attention' (C)."""

    drop_value: str = "zero"
    """Replacement for dropped tokens: 'zero', 'mean', or 'noise'."""

    top_k_ratio: float = 0.3
    """Variant C only: fraction of tokens to drop (0.0-1.0)."""

    static_attention: bool = True
    """Variant C only: reuse first-step attention weights for all steps."""

    clamp_ratio: float = 0.3
    """Maximum deviation ratio for velocity clamping. 0 disables clamping."""

    verbose: bool = False
    """Log per-step delta and drop statistics."""


def main(config: TokenDropServerConfig):
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)
    log.info("Starting GR00T Token-Drop CD inference server...")
    log.info("  Embodiment tag: %s", config.embodiment_tag)
    log.info("  Model path: %s", config.model_path)
    log.info("  Device: %s", config.device)
    log.info("  Variant: %s", config.variant)
    log.info("  Alpha: %.2f", config.alpha)
    log.info("  Drop value: %s", config.drop_value)
    log.info("  Clamp ratio: %.2f", config.clamp_ratio)
    if config.variant == "attention":
        log.info("  Top-K ratio: %.2f", config.top_k_ratio)
        log.info("  Static attention: %s", config.static_attention)
    log.info("  Host: %s", config.host)
    log.info("  Port: %d", config.port)

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = TokenDropContrastivePolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        alpha=config.alpha,
        variant=DropVariant(config.variant),
        drop_value=config.drop_value,
        top_k_ratio=config.top_k_ratio,
        static_attention=config.static_attention,
        clamp_ratio=config.clamp_ratio,
        verbose=config.verbose,
    )

    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    try:
        server.run()
    except KeyboardInterrupt:
        log.info("Shutting down server...")


if __name__ == "__main__":
    config = tyro.cli(TokenDropServerConfig)
    main(config)
