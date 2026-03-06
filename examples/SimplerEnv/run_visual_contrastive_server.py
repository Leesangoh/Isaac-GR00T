"""GR00T inference server with Visual Contrastive Decoding (VCD).

Drop-in replacement for gr00t/eval/run_gr00t_server.py that uses
VisualContrastivePolicy (noisy-image VLM as amateur).

Usage:
    uv run python examples/SimplerEnv/run_visual_contrastive_server.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --use-sim-policy-wrapper \
        --alpha 1.0 \
        --noise-std 0.5
"""

from dataclasses import dataclass
import logging
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.server_client import PolicyServer
from gr00t.policy.visual_contrastive_policy import VisualContrastivePolicy
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class VCDServerConfig:
    """Configuration for GR00T VCD inference server."""

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

    # VCD configs
    alpha: float = 1.0
    """Contrastive amplification factor. 0.0 = vanilla."""

    noise_std: float = 0.5
    """Gaussian noise std added to normalized pixel_values (typical range 0.1-1.0)."""

    clamp_ratio: float = 0.3
    """Maximum deviation ratio for velocity clamping. 0 disables clamping."""

    verbose: bool = False
    """Log per-step delta and noise statistics."""


def main(config: VCDServerConfig):
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)
    log.info("Starting GR00T VCD inference server...")
    log.info("  Embodiment tag: %s", config.embodiment_tag)
    log.info("  Model path: %s", config.model_path)
    log.info("  Device: %s", config.device)
    log.info("  Alpha: %.2f", config.alpha)
    log.info("  Noise std: %.3f", config.noise_std)
    log.info("  Clamp ratio: %.2f", config.clamp_ratio)
    log.info("  Host: %s", config.host)
    log.info("  Port: %d", config.port)

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = VisualContrastivePolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        alpha=config.alpha,
        noise_std=config.noise_std,
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
    config = tyro.cli(VCDServerConfig)
    main(config)
