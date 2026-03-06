"""GR00T inference server with Restart Sampling.

Drop-in replacement for gr00t/eval/run_gr00t_server.py that uses
RestartSamplingPolicy (stochastic restart cycles for improved sample quality).

Usage:
    uv run python examples/SimplerEnv/run_restart_server.py \
        --model-path nvidia/GR00T-N1.6-fractal \
        --embodiment-tag OXE_GOOGLE \
        --use-sim-policy-wrapper \
        --t-restart 0.6 \
        --t-back 0.3 \
        --restart-K 1 \
        --steps-restart 3 \
        --total-steps 10 \
        --noise-method sdedit
"""

from dataclasses import dataclass
import logging
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.restart_sampling_policy import RestartSamplingPolicy
from gr00t.policy.server_client import PolicyServer
import tyro


DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class RestartServerConfig:
    """Configuration for GR00T Restart Sampling inference server."""

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

    # Restart Sampling configs
    t_restart: float = 0.6
    """Restart point in [0, 1] (0=noise, 1=data)."""

    t_back: float = 0.3
    """Re-noise target (must be < t_restart)."""

    restart_K: int = 1
    """Number of restart cycles. 0 = vanilla ODE."""

    steps_restart: int = 3
    """ODE steps per restart cycle."""

    total_steps: int = 10
    """Total NFE budget."""

    noise_method: str = "sdedit"
    """Re-noise method: 'sdedit', 'interpolation', or 'scaled'."""

    verbose: bool = False
    """Log per-step statistics."""


def main(config: RestartServerConfig):
    if config.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    log = logging.getLogger(__name__)
    log.info("Starting GR00T Restart Sampling inference server...")
    log.info("  Embodiment tag: %s", config.embodiment_tag)
    log.info("  Model path: %s", config.model_path)
    log.info("  Device: %s", config.device)
    log.info("  t_restart: %.2f", config.t_restart)
    log.info("  t_back: %.2f", config.t_back)
    log.info("  restart_K: %d", config.restart_K)
    log.info("  steps_restart: %d", config.steps_restart)
    log.info("  total_steps: %d", config.total_steps)
    log.info("  noise_method: %s", config.noise_method)
    log.info("  Host: %s", config.host)
    log.info("  Port: %d", config.port)

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    policy = RestartSamplingPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        t_restart=config.t_restart,
        t_back=config.t_back,
        restart_K=config.restart_K,
        steps_restart=config.steps_restart,
        total_steps=config.total_steps,
        noise_method=config.noise_method,
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
    config = tyro.cli(RestartServerConfig)
    main(config)
