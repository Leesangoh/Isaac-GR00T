"""GR00T + Intent Passthrough server for client-side cerebellum (Option 3).

Runs GR00T and returns raw action chunks + intent vectors in info dict.
No correction is applied server-side. The client loads the cerebellum locally.

Usage:
    LD_PRELOAD=.venv/lib/libglibc_compat.so \
    uv run python cerebellar_correction/eval/run_intent_server.py \
        --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
        --embodiment-tag OXE_WIDOWX
"""

from dataclasses import dataclass
import logging
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import tyro

from cerebellar_correction.policy.intent_passthrough_policy import IntentPassthroughPolicyWrapper


log = logging.getLogger(__name__)


@dataclass
class IntentServerConfig:
    """Configuration for GR00T + Intent Passthrough server."""

    model_path: str = "/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge"
    """Path to the fine-tuned GR00T checkpoint."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag for the robot type."""

    device: str = "cuda"
    """Device to run the model on."""

    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = 5555
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input and output validation."""

    save_attention_map: bool = False
    """Save cross-attention heatmaps from the DiT action head."""

    attention_map_dir: str = "./attention_maps"
    """Directory to save attention map images."""


def main(config: IntentServerConfig):
    logging.basicConfig(level=logging.INFO)

    log.info("Starting GR00T + Intent Passthrough server (client-side cerebellum)...")
    log.info("  Model path: %s", config.model_path)
    log.info("  Embodiment tag: %s", config.embodiment_tag)
    log.info("  Device: %s", config.device)
    log.info("  Host: %s:%d", config.host, config.port)

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    gr00t_policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        save_attention_map=config.save_attention_map,
        attention_map_dir=config.attention_map_dir,
    )

    sim_policy = Gr00tSimPolicyWrapper(gr00t_policy)

    policy = IntentPassthroughPolicyWrapper(
        policy=sim_policy,
        groot_model=gr00t_policy.model,
        device=config.device,
    )

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
    config = tyro.cli(IntentServerConfig)
    main(config)
