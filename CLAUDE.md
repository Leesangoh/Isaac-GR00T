# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Isaac GR00T N1.6 — a 3B-parameter open vision-language-action (VLA) model for generalized robot manipulation. Architecture: Eagle VLM backbone (Cosmos-Reason-2B variant) + 32-layer DiT action head with flow matching. Predicts state-relative action chunks from multimodal input (language + images).

## Environment Setup

- **Python 3.10** (strict), **uv >= 0.8.4** for dependency management
- CUDA 12.4 recommended; flash-attn 2.7.4 required
- `LD_PRELOAD=.venv/lib/libglibc_compat.so` may be needed for flash_attn on some systems

```bash
uv sync --python 3.10
uv pip install -e .
```

## Common Commands

```bash
# Lint & format
ruff format .
ruff check --fix .

# Finetune (simple)
uv run python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path <DATA> --embodiment-tag <TAG> --output-dir <OUT>

# Finetune (full control via YAML config)
uv run python gr00t/experiment/launch_train.py --load-config-path <YAML>

# Inference
uv run python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.6-3B --dataset-path <DATA> \
  --embodiment-tag <TAG> --action-horizon 8

# Policy server
uv run python gr00t/eval/run_gr00t_server.py \
  --embodiment-tag <TAG> --model-path <CHECKPOINT>

# Open-loop evaluation
uv run python gr00t/eval/open_loop_eval.py \
  --dataset-path <DATA> --embodiment-tag <TAG> --model-path <CHECKPOINT>

# Benchmark inference latency
uv run python scripts/deployment/benchmark_inference.py
```

There is no test suite; CI runs `ruff format --check .` and `ruff check .` only.

## Code Style

- **Ruff** with 100-char line length, double quotes, Python 3.10 target
- Rules: E (errors), F (pyflakes), I (isort). E501 (line length) is ignored.
- `__init__.py` files ignore F401 (unused imports)
- Excluded from linting: `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/`, `external_dependencies/`

## Architecture

### Data Flow
1. **Dataset** (`gr00t/data/`) — LeRobot V2 format episodes loaded via `lerobot_episode_loader.py`. `VLAStepData` (defined in `types.py`) is the core per-step type holding images, states, actions, and language instructions.
2. **Collation** (`gr00t/data/collator/`) — Batches `VLAStepData` with Eagle processor for vision encoding.
3. **Model** (`gr00t/model/gr00t_n1d6/`) — `Gr00tN1d6ActionHead` combines Eagle backbone → DiT action head. Processing in `processing_gr00t_n1d6.py`.
4. **Training** (`gr00t/experiment/`) — `Gr00tTrainer` (custom HuggingFace Trainer) with DeepSpeed, wandb, profiling. Entry points: `launch_finetune.py` (simple) and `launch_train.py` (full).
5. **Inference** (`gr00t/policy/`) — `Gr00tPolicy` wraps model for observation→action. Supports server-client via REST API (`server_client.py`).

### Key Module Paths
- **Eagle backbone**: `gr00t/model/modules/eagle_backbone.py` → vendor model in `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/`
- **DiT action head**: `gr00t/model/modules/dit.py` (32-layer diffusion transformer with VLM cross-attention)
- **Config system**: `gr00t/configs/base_config.py` (top-level `Config` dataclass), `gr00t/configs/model/gr00t_n1d6.py` (model hyperparams), `gr00t/configs/finetune_config.py` (user-facing finetune params), `gr00t/configs/training/training_config.py` (LR, warmup, batch size, DeepSpeed)
- **Embodiment tags**: `gr00t/data/embodiment_tags.py` — enum of supported robots (LIBERO_PANDA, OXE_WIDOWX, UNITREE_G1, GR1, DROID, etc.)
- **Per-embodiment data configs**: `gr00t/configs/data/` — modality configs specifying delta_indices and modality_keys per robot

### Training
- Distributed via `torchrun` + DeepSpeed (stage 2/3 configs in `gr00t/configs/deepspeed/`)
- LoRA/adapter support via `peft`
- Experiment tracking with `wandb`
- CLI parsing via `tyro`

### External Dependencies
Git submodules in `external_dependencies/`: LIBERO, SimplerEnv, robocasa (benchmarks), GR00T-WholeBodyControl.
