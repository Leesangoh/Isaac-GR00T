from pathlib import Path
import shutil

from transformers import TrainerCallback
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


class CheckpointFormatCallback(TrainerCallback):
    """This callback format checkpoint to make them standalone. For now, it copies all config
    files to /checkpoint-{step}/experiment_cfg/:
    - conf.yaml
    - initial_actions.npz
    - metadata.json
    """

    def __init__(
        self, run_name: str, exp_cfg_dir: Path | None = None, processor_dir: Path | None = None
    ):
        """
        Args:
            run_name: Name of the experiment run
            exp_cfg_dir: Path to the directory containing all experiment metadata
        """
        self.exp_cfg_dir = exp_cfg_dir
        self.processor_dir = processor_dir

    def on_save(self, args, state, control, **kwargs):
        """Called after the trainer saves a checkpoint."""
        if state.is_world_process_zero:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"

            # Copy experiment config directory if provided
            if self.exp_cfg_dir is not None:
                exp_cfg_dst = checkpoint_dir / self.exp_cfg_dir.name
                if self.exp_cfg_dir.exists():
                    print(
                        f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                    )
                    shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

            # Copy processor directory if provided
            if self.processor_dir is not None:
                if self.processor_dir.exists():
                    print(f"Copying processor directory {self.processor_dir} to {checkpoint_dir}")
                    shutil.copytree(self.processor_dir, checkpoint_dir, dirs_exist_ok=True)

            # Copy wandb_config.json if provided
            wandb_config_src = Path(args.output_dir) / "wandb_config.json"
            wandb_config_dst = checkpoint_dir / "wandb_config.json"
            if wandb_config_src.exists():
                print(f"Copying wandb_config.json from {wandb_config_src} to {wandb_config_dst}")
                shutil.copy2(wandb_config_src, wandb_config_dst)


class ModelOnlyCheckpointCallback(TrainerCallback):
    """Save model-only checkpoints (no optimizer/scheduler state) at every ``model_save_steps``
    steps into a separate ``model-checkpoints/`` subdirectory under the output dir.

    These are independent from the rolling full checkpoints managed by ``save_total_limit``,
    so they are never deleted automatically.

    When a full checkpoint coincides with a model-only step (i.e. step is a multiple of both
    ``save_steps`` and ``model_save_steps``), the full checkpoint is protected from deletion
    by being copied into the model-only directory as well.
    """

    def __init__(self, model_save_steps: int = 10000, exp_cfg_dir: Path | None = None):
        self.model_save_steps = model_save_steps
        self.exp_cfg_dir = exp_cfg_dir

    def on_save(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if step % self.model_save_steps != 0:
            return

        if not state.is_world_process_zero:
            return

        output_dir = Path(args.output_dir)
        model_ckpt_dir = output_dir / "model-checkpoints" / f"checkpoint-{step}"
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights only using save_pretrained (works with DeepSpeed)
        if model is not None:
            unwrapped = model.module if hasattr(model, "module") else model
            unwrapped.save_pretrained(model_ckpt_dir)
        else:
            # Fallback: copy non-optimizer files from full checkpoint
            full_ckpt_dir = output_dir / f"checkpoint-{step}"
            if full_ckpt_dir.exists():
                skip_patterns = (
                    "optimizer", "scheduler", "rng_state", "training_args",
                    "zero_pp_rank", "optim_states", "global_step",
                )
                for item in full_ckpt_dir.iterdir():
                    if any(p in item.name for p in skip_patterns):
                        continue
                    dst = model_ckpt_dir / item.name
                    if item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, dst)

        # Copy experiment config if available
        if self.exp_cfg_dir is not None and self.exp_cfg_dir.exists():
            exp_cfg_dst = model_ckpt_dir / self.exp_cfg_dir.name
            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

        print(f"[ModelOnlyCheckpoint] Saved model-only checkpoint at step {step} to {model_ckpt_dir}")


class BestMetricCheckpointCallback(TrainerCallback):
    """This callback saves the best checkpoint based on the metric."""

    def __init__(
        self, metric_name: str, greater_is_better: bool = True, exp_cfg_dir: Path | None = None
    ):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best_metric = -float("inf") if greater_is_better else float("inf")
        self.exp_cfg_dir = exp_cfg_dir
        self._best_checkpoint_dir = None

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics,
        model,
        **kwargs,
    ):
        if state.is_world_process_zero and metrics is not None:
            current_metric = metrics.get(self.metric_name, None)
            if current_metric is not None:
                is_better = (
                    self.greater_is_better
                    if current_metric > self.best_metric
                    else not self.greater_is_better
                )
                if is_better:
                    self.best_metric = current_metric
                    best_checkpoint_dir = (
                        Path(args.output_dir)
                        / f"checkpoint-{state.global_step}-best-{self.metric_name}_{current_metric}"
                    )
                    best_checkpoint_dir.mkdir(exist_ok=True)
                    model.save_pretrained(best_checkpoint_dir)
                    # Copy experiment config directory if provided
                    if self.exp_cfg_dir is not None:
                        exp_cfg_dst = best_checkpoint_dir / self.exp_cfg_dir.name
                        if self.exp_cfg_dir.exists():
                            print(
                                f"Copying experiment config directory {self.exp_cfg_dir} to {exp_cfg_dst}"
                            )
                            shutil.copytree(self.exp_cfg_dir, exp_cfg_dst, dirs_exist_ok=True)

                    print(
                        f"Best checkpoint saved to {best_checkpoint_dir} with metric {self.metric_name} = {current_metric}"
                    )

                    if (
                        self._best_checkpoint_dir is not None
                        and Path(self._best_checkpoint_dir).exists()
                    ):
                        shutil.rmtree(self._best_checkpoint_dir)

                    self._best_checkpoint_dir = str(best_checkpoint_dir)
