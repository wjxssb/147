# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import pprint
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import pytorch_lightning as pl
import torch
import torch.serialization
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

# PyTorch 2.6 weights_only security allowlist for Hydra configs saved in Lightning checkpoints
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([ListConfig, DictConfig])
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from emg2qwerty import transforms, utils
from emg2qwerty.transforms import Transform


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def main(config: DictConfig):
    log.info(f"\nConfig:\n{OmegaConf.to_yaml(config)}")

    # Add working dir to PYTHONPATH
    working_dir = get_original_cwd()
    python_paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if working_dir not in python_paths:
        python_paths.append(working_dir)
        os.environ["PYTHONPATH"] = os.pathsep.join(python_paths)

    # Seed for determinism. This seeds torch, numpy and python random modules
    # taking global rank into account (for multi-process distributed setting).
    # Additionally, this auto-adds a worker_init_fn to train_dataloader that
    # initializes the seed taking worker_id into account per dataloading worker
    # (see `pl_worker_init_fn()`).
    pl.seed_everything(config.seed, workers=True)

    # Helper to instantiate full paths for dataset sessions
    def _full_session_paths(dataset: ListConfig) -> list[Path]:
        sessions = [session["session"] for session in dataset]
        return [
            Path(config.dataset.root).joinpath(f"{session}.hdf5")
            for session in sessions
        ]

    # Helper to instantiate transforms
    def _build_transform(configs: Sequence[DictConfig]) -> Transform[Any, Any]:
        return transforms.Compose([instantiate(cfg) for cfg in configs])

    # Instantiate LightningModule
    log.info(f"Instantiating LightningModule {config.module}")
    module = instantiate(
        config.module,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
        decoder=config.decoder,
        _recursive_=False,
    )
    if config.checkpoint is not None:
        log.info(f"Loading module from checkpoint {config.checkpoint}")
        module = type(module).load_from_checkpoint(
            config.checkpoint,
            optimizer=config.optimizer,
            lr_scheduler=config.lr_scheduler,
            decoder=config.decoder,
            weights_only=False,
        )

    # Instantiate LightningDataModule
    log.info(f"Instantiating LightningDataModule {config.datamodule}")
    datamodule = instantiate(
        config.datamodule,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        train_sessions=_full_session_paths(config.dataset.train),
        val_sessions=_full_session_paths(config.dataset.val),
        test_sessions=_full_session_paths(config.dataset.test),
        train_transform=_build_transform(config.transforms.train),
        val_transform=_build_transform(config.transforms.val),
        test_transform=_build_transform(config.transforms.test),
        _convert_="object",
    )

    # Instantiate callbacks
    callback_configs = config.get("callbacks", [])
    callbacks = [instantiate(cfg) for cfg in callback_configs]

    # Initialize trainer
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=callbacks,
    )

    if config.train:
        # Check if a past checkpoint exists to resume training from
        checkpoint_dir = Path.cwd().joinpath("checkpoints")
        resume_from_checkpoint = utils.get_last_checkpoint(checkpoint_dir)
        if resume_from_checkpoint is not None:
            log.info(f"Resuming training from checkpoint {resume_from_checkpoint}")

        # Train
        trainer.fit(module, datamodule, ckpt_path=resume_from_checkpoint)

        # Load best checkpoint
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", None)
        if best_path and os.path.isfile(best_path):
            log.info(f"Loading best checkpoint from {best_path}")
            module = type(module).load_from_checkpoint(
                best_path,
                optimizer=config.optimizer,
                lr_scheduler=config.lr_scheduler,
                decoder=config.decoder,
                weights_only=False,
            )

    # Validate on the best checkpoint (if training), or on the loaded checkpoint
    val_metrics = trainer.validate(module, datamodule)

    # Evaluate on test set ONLY when explicitly requested (to prevent test set leakage during development)
    test_metrics = None
    if config.get("evaluate_test", False):
        log.info("Evaluating on frozen test set...")
        test_metrics = trainer.test(module, datamodule)

    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        log.info(f"Peak VRAM: {peak_vram_mb:.2f} MB")

    results = {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_checkpoint": getattr(trainer.checkpoint_callback, "best_model_path", None),
        "peak_vram_mb": peak_vram_mb,
    }
    pprint.pprint(results, sort_dicts=False)

    try:
        output_dir = Path(trainer.default_root_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        if hasattr(module, "val_predictions") and module.val_predictions:
            with open(output_dir / "val_predictions.json", "w") as f:
                json.dump(module.val_predictions, f, indent=2)
            log.info(f"Saved {len(module.val_predictions)} validation predictions to val_predictions.json")
        if hasattr(module, "test_predictions") and module.test_predictions:
            with open(output_dir / "test_predictions.json", "w") as f:
                json.dump(module.test_predictions, f, indent=2)
            log.info(f"Saved {len(module.test_predictions)} test predictions to test_predictions.json")
    except Exception as e:
        log.warning(f"Could not save results.json: {e}")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("cpus_per_task", utils.cpus_per_task)
    main()
