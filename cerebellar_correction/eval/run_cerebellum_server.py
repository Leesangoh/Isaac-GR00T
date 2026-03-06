"""GR00T + Intent-Conditioned Cerebellar Correction inference server.

Same interface as run_gr00t_server.py. The cerebellum hooks GR00T's backbone
to extract intent vectors and applies prediction-error correction to actions.

Usage:
    LD_PRELOAD=.venv/lib/libglibc_compat.so \
    .venv/bin/python cerebellar_correction/eval/run_cerebellum_server.py \
        --model-path /mnt/md1/solee/checkpoints/GR00T-N1.6-bridge \
        --embodiment-tag OXE_WIDOWX \
        --cerebellum-ckpt checkpoints/cerebellum_intent \
        --correction-alpha 1.0
"""

from dataclasses import dataclass
import logging
import os

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
from gr00t.policy.server_client import PolicyServer
import tyro

from cerebellar_correction.policy.cerebellum_policy import IntentCerebellumPolicyWrapper


DEFAULT_MODEL_SERVER_PORT = 5555

log = logging.getLogger(__name__)


@dataclass
class CerebellumServerConfig:
    """Configuration for GR00T + Intent Cerebellum inference server."""

    model_path: str = "/mnt/md1/solee/checkpoints/GR00T-N1.6-bridge"
    """Path to the fine-tuned GR00T checkpoint."""

    embodiment_tag: EmbodimentTag = EmbodimentTag.NEW_EMBODIMENT
    """Embodiment tag for the robot type."""

    device: str = "cuda"
    """Device to run the model on."""

    # Cerebellum configs
    cerebellum_ckpt: str = "checkpoints/cerebellum_intent"
    """Path to cerebellum checkpoint directory (with phase1/ and phase2/ subdirs)."""

    correction_alpha: float = 1.0
    """Correction strength (0=no correction, 1=full correction)."""

    max_correction: float = 0.15
    """Maximum correction magnitude per action dimension."""

    # Server configs
    host: str = "0.0.0.0"
    """Host address for the server."""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server."""

    strict: bool = True
    """Whether to enforce strict input and output validation."""

    save_attention_map: bool = False
    """Save cross-attention heatmaps from the DiT action head."""

    attention_map_dir: str = "./attention_maps"
    """Directory to save attention map images."""


def main(config: CerebellumServerConfig):
    logging.basicConfig(level=logging.INFO)

    log.info("Starting GR00T + Intent Cerebellum server...")
    log.info("  Model path: %s", config.model_path)
    log.info("  Embodiment tag: %s", config.embodiment_tag)
    log.info(
        "  Cerebellum: %s (alpha=%.2f, max_corr=%.3f)",
        config.cerebellum_ckpt,
        config.correction_alpha,
        config.max_correction,
    )
    log.info("  Device: %s", config.device)
    log.info("  Host: %s:%d", config.host, config.port)

    if config.model_path.startswith("/") and not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Model path {config.model_path} does not exist")

    # 1. Load GR00T policy
    gr00t_policy = Gr00tPolicy(
        embodiment_tag=config.embodiment_tag,
        model_path=config.model_path,
        device=config.device,
        strict=config.strict,
        save_attention_map=config.save_attention_map,
        attention_map_dir=config.attention_map_dir,
    )

    # 2. Sim policy wrapper
    sim_policy = Gr00tSimPolicyWrapper(gr00t_policy)

    # 3. Intent cerebellum wrapper (hooks GR00T backbone + corrects actions)
    policy = IntentCerebellumPolicyWrapper(
        policy=sim_policy,
        groot_model=gr00t_policy.model,
        cerebellum_ckpt=config.cerebellum_ckpt,
        device=config.device,
        correction_alpha=config.correction_alpha,
        max_correction=config.max_correction,
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
    config = tyro.cli(CerebellumServerConfig)
    main(config)
