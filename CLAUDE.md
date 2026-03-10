# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NVIDIA Isaac GR00T N1.6 is a 3B-parameter vision-language-action (VLA) foundation model for humanoid robot control. It takes multimodal input (language instructions + camera images) and outputs action predictions via a diffusion transformer head. The model supports cross-embodiment training across 30+ robot platforms.

## Build & Development Commands

**Python 3.10 is required.** Uses `uv` (v0.8.4+) as the package manager.

```bash
# Install (editable mode with dev tools)
uv pip install -e .[dev]

# Lint and format
ruff format .
ruff check --fix .

# CI runs these checks (must pass before PR merge)
ruff check .
ruff format --check .

# Run tests
pytest -v tests/a/b_test.py

# Finetuning
python gr00t/experiment/launch_finetune.py --help

# Standalone inference
python scripts/deployment/standalone_inference_script.py \
    --model-path nvidia/GR00T-N1.6-3B \
    --dataset-path demo_data/gr1.PickNPlace \
    --embodiment-tag GR1

# Policy server
python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag GR1 \
    --model-path nvidia/GR00T-N1.6-3B \
    --device cuda:0
```

## Code Style

- **Formatter/Linter**: `ruff` (line-length=100, target py310)
- **Lint rules**: E (pycodestyle), F (pyflakes), I (isort). E501 (long lines) is ignored.
- **Quotes**: double. **Indent**: spaces.
- **isort**: case-insensitive, `lines-after-imports = 2`, `combine-as-imports = true`, `force-sort-within-sections = true`.
- `__init__.py` files may have unused imports (F401 ignored).
- Ruff excludes: `gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2/` and `external_dependencies/`.
- CI only runs lint checks (`ruff check .` and `ruff format --check .`) — no automated test suite.

## Architecture

### Core Package Structure (`gr00t/`)

- **`model/`** — Model implementations
  - `gr00t_n1d6/` — Main GR00T N1.6 model (HuggingFace `PreTrainedModel`)
  - `modules/` — Neural network building blocks: DiT (diffusion transformer), Eagle backbone (VLM), embodiment-conditioned MLPs, flow matching
  - `modules/nvidia/` — Proprietary NVIDIA VLM (Cosmos-Reason-2B). Excluded from linting.

- **`data/`** — Data loading and processing pipeline
  - `types.py` — Core data structure `VLAStepData` (images, states, actions, text, embodiment tag)
  - `embodiment_tags.py` — `EmbodimentTag` enum for all supported robots
  - `dataset/` — Dataset implementations (LeRobot V2 format, sharded loading, episode-based)
  - `state_action/` — Action normalization, chunking, pose transforms
  - `collator/` — Batch collation with padding/masking

- **`configs/`** — Configuration system (dataclasses + `tyro` CLI parsing)
  - `base_config.py` — Top-level `Config` with model, data, training sub-configs
  - `data/embodiment_configs.py` — Pre-registered modality configs per robot
  - `model/gr00t_n1d6.py` — Model architecture hyperparameters
  - `training/training_config.py` — Training hyperparameters (DeepSpeed, mixed precision)

- **`policy/`** — Inference API
  - `gr00t_policy.py` — `Gr00tPolicy`: main inference class. Takes observations dict → returns actions dict
  - `server_client.py` — REST API for distributed inference (policy on GPU machine, robot on another)

- **`experiment/`** — Training pipeline
  - `launch_finetune.py` — Simple finetuning entry point
  - `launch_train.py` — Full training with advanced options
  - `trainer.py` — Custom HuggingFace `Trainer` subclass with profiling and DeepSpeed integration

- **`eval/`** — Evaluation framework
  - `open_loop_eval.py` — Offline evaluation (predicted vs ground truth actions)
  - `sim/` — Simulation wrappers (LIBERO, BEHAVIOR, RoboCasa, etc.)
  - `real_robot/` — Real robot evaluation (SO-100)

### Key Architectural Patterns

1. **Embodiment system**: Each robot type has an `EmbodimentTag` and a `ModalityConfig` defining its observation/action spaces, temporal sampling (`delta_indices`), and action representations (relative/absolute, EEF/joint). New embodiments register configs via `register_modality_config()`.

2. **Data flow**: Raw data (LeRobot V2 format) → `VLAStepData` → `Gr00tN1d6Processor` (normalization via `StateActionProcessor` using dataset statistics) → `Gr00tN1d6DataCollator` (batching/padding) → model input tensors.

3. **Model architecture**: Eagle VLM backbone (vision + language encoding) → Diffusion Transformer action head (32-layer DiT with flow matching) → action predictions. Embodiment-conditioned MLPs route state/action encoding per robot type.

4. **Policy inference**: `Gr00tPolicy` wraps the model for deployment. Observations are numpy dicts with shape `(B, T, ...)`. The server-client pattern in `server_client.py` enables remote GPU inference via REST API.

5. **Training**: HuggingFace Trainer + DeepSpeed ZeRO. Configuration via `tyro` CLI. Supports LoRA (PEFT), gradient checkpointing, bfloat16, and WandB logging.

### Video-Depth-Anything (`Video-Depth-Anything/`)

Temporally consistent monocular depth estimation for video (CVPR 2025 Highlight, ByteDance). Used to generate depth maps from sim evaluation videos.

**Setup & Usage:**
```bash
cd Video-Depth-Anything
# Dependencies conflict with main gr00t env (torch 2.1.1 vs 2.7.1) — use separate venv
pip install -r requirements.txt

# Weights are in checkpoints/ (Small model: 28.4M params, 112MB)
# Run depth estimation (Small model, fastest):
python3 run.py --input_video <input.mp4> --output_dir ./outputs --encoder vits

# Encoder options: vits (28.4M) | vitb (113.1M) | vitl (381.8M)
# Add --metric for metric depth, --grayscale for grayscale output
# Add --save_npz to save raw depth arrays
```

**Architecture:** DINOv2 backbone + DPT decoder with temporal attention (`dpt_temporal.py`). Processes full video at once for frame-to-frame consistency. Outputs `_src.mp4` (original) and `_vis.mp4` (colorized depth map) side by side.

**Key files:**
- `run.py` — Main inference entry point
- `video_depth_anything/video_depth.py` — Core model class `VideoDepthAnything`
- `video_depth_anything/dpt_temporal.py` — DPT head with temporal attention modules
- `utils/dc_utils.py` — Video I/O utilities (`read_video_frames`, `save_video`)

## Examples & Benchmarks

The `examples/` directory contains self-contained benchmark integrations (LIBERO, BEHAVIOR, SimplerEnv, RoboCasa, SO100, DROID, PointNav, GR00T-WholeBodyControl). Each has its own README with setup instructions.

## Key Dependencies

PyTorch 2.7.1, Transformers 4.51.3, flash-attn 2.7.4, DeepSpeed 0.17.6, PEFT 0.17.1, torchcodec (video decoding), tyro (CLI), WandB (experiment tracking).
