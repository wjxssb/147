# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    CNNBiGRUEncoder,
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.train_transform,
                        window_length=self.window_length,
                        padding=self.padding,
                        jitter=True,
                    )
                    for hdf5_path in self.train_sessions
                ]
            )
            self.val_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.val_transform,
                        window_length=self.window_length,
                        padding=self.padding,
                        jitter=False,
                    )
                    for hdf5_path in self.val_sessions
                ]
            )
        if stage in (None, "test"):
            self.test_dataset = ConcatDataset(
                [
                    WindowedEMGDataset(
                        hdf5_path,
                        transform=self.test_transform,
                        # Feed the entire session at once without windowing/padding
                        # at test time for more realism
                        window_length=None,
                        padding=(0, 0),
                        jitter=False,
                    )
                    for hdf5_path in self.test_sessions
                ]
            )

    def train_dataloader(self) -> DataLoader:
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            self.setup("fit")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            self.setup("test")
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        num_bands: int = 2,
        electrode_channels: int = 16,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_bands = num_bands
        self.electrode_channels = electrode_channels

        num_features = self.num_bands * mlp_features[-1]

        # Model
        # inputs: (T, N, bands, electrode_channels, freq)
        self.model = nn.Sequential(
            # (T, N, bands, C, freq)
            SpectrogramNorm(channels=self.num_bands * self.electrode_channels),
            # (T, N, bands, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.num_bands,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion with zero_infinity=True
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        self.val_predictions: list[dict[str, str]] = []
        self.test_predictions: list[dict[str, str]] = []

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        # CTC Alignment check
        illegal_count = 0
        for i in range(N):
            seq = targets[: target_lengths[i], i]
            dup = (seq[1:] == seq[:-1]).sum().item() if len(seq) > 1 else 0
            min_req = target_lengths[i].item() + dup
            if emission_lengths[i].item() < min_req:
                illegal_count += 1
        if illegal_count > 0:
            self.log(f"{phase}/illegal_ctc_samples", float(illegal_count), batch_size=N, sync_dist=True)

        if phase == "test":
            loss = self.ctc_loss(
                log_probs=emissions.cpu(),
                targets=targets.cpu().transpose(0, 1),
                input_lengths=emission_lengths.cpu(),
                target_lengths=target_lengths.cpu(),
            ).to(emissions.device)
        else:
            loss = self.ctc_loss(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)
            if phase == "val":
                self.val_predictions.append({"hyp": predictions[i].text, "ref": target.text})
            elif phase == "test":
                self.test_predictions.append({"hyp": predictions[i].text, "ref": target.text})

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_predictions = []

    def on_test_epoch_start(self) -> None:
        self.test_predictions = []

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


class CNNBiGRUCTCModule(pl.LightningModule):
    """CNN-BiGRU-CTC Lightning Module for sEMG Keystroke Prediction.

    Combines multi-band rotation invariant spatial feature extraction with a
    hybrid 1D-CNN + Bidirectional GRU temporal encoder.
    Maintains exact temporal length (stride=1, padding=1), ensuring emission_lengths == input_lengths.
    Includes explicit CTC length validation (T_out >= L + R) and illegal sample logging.

    Args:
        in_features (int): Input features per band to MLP (e.g. freq_bins * channels).
        mlp_features (Sequence[int]): Hidden dimensions for spatial MLP. (e.g. [384])
        hidden_size (int): GRU hidden size per direction. (default: 256)
        num_layers (int): Number of stacked GRU layers. (default: 2)
        dropout (float): Dropout probability. (default: 0.2)
        kernel_size (int): Kernel size for 1D convolution. (default: 3)
        num_bands (int): Number of sEMG bands (2 for dual-wrist, 1 for single-wrist). (default: 2)
        electrode_channels (int): Electrodes per band (16, 8, or 4). (default: 16)
        optimizer (DictConfig): Optimizer configuration.
        lr_scheduler (DictConfig): LR scheduler configuration.
        decoder (DictConfig): CTC decoder configuration.
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
        kernel_size: int = 3,
        num_bands: int = 2,
        electrode_channels: int = 16,
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.num_bands = num_bands
        self.electrode_channels = electrode_channels

        num_features = self.num_bands * mlp_features[-1]

        # Frontend: Normalization + Spatial Rotation-Invariant MLP
        self.frontend = nn.Sequential(
            SpectrogramNorm(channels=self.num_bands * self.electrode_channels),
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.num_bands,
            ),
            nn.Flatten(start_dim=2),  # (T, N, num_features)
        )

        # Temporal Encoder: CNN-BiGRU
        self.encoder = CNNBiGRUEncoder(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            kernel_size=kernel_size,
        )

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion with zero_infinity=True
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class, zero_infinity=True)

        # Decoder
        self.decoder = instantiate(decoder) if decoder is not None else None

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )
        self.val_predictions: list[dict[str, str]] = []
        self.test_predictions: list[dict[str, str]] = []

        self.num_illegal_alignments = 0

    def forward(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass through frontend -> CNN-BiGRU encoder -> classifier."""
        features = self.frontend(inputs)  # (T, N, num_features)
        encoded = self.encoder(features, input_lengths=input_lengths)  # (T, N, num_features)
        emissions = self.classifier(encoded)  # (T, N, num_classes)
        return emissions

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs, input_lengths=input_lengths)

        # For CNN-BiGRU with padding=1, emissions.shape[0] == inputs.shape[0], T_diff = 0
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        # Rigorous CTC Alignment constraint verification: T_out >= L + R
        illegal_batch_count = 0
        for i in range(N):
            tgt_len = target_lengths[i].item()
            seq = targets[:tgt_len, i]
            # Consecutive duplicate pairs require intervening blank
            dup_pairs = (seq[1:] == seq[:-1]).sum().item() if tgt_len > 1 else 0
            min_required = tgt_len + dup_pairs
            if emission_lengths[i].item() < min_required:
                illegal_batch_count += 1

        if illegal_batch_count > 0:
            self.num_illegal_alignments += illegal_batch_count
            self.log(
                f"{phase}/illegal_ctc_samples",
                float(illegal_batch_count),
                batch_size=N,
                sync_dist=True,
            )

        if phase == "test":
            loss = self.ctc_loss(
                log_probs=emissions.cpu(),
                targets=targets.cpu().transpose(0, 1),
                input_lengths=emission_lengths.cpu(),
                target_lengths=target_lengths.cpu(),
            ).to(emissions.device)
        else:
            loss = self.ctc_loss(
                log_probs=emissions,  # (T, N, num_classes)
                targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
                input_lengths=emission_lengths,  # (N,)
                target_lengths=target_lengths,  # (N,)
            )

        # Decode emissions if decoder configured
        if self.decoder is not None:
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )
            metrics = self.metrics[f"{phase}_metrics"]
            targets_np = targets.detach().cpu().numpy()
            target_lengths_np = target_lengths.detach().cpu().numpy()
            for i in range(N):
                target = LabelData.from_labels(targets_np[: target_lengths_np[i], i])
                metrics.update(prediction=predictions[i], target=target)
                if phase == "val":
                    self.val_predictions.append({"hyp": predictions[i].text, "ref": target.text})
                elif phase == "test":
                    self.test_predictions.append({"hyp": predictions[i].text, "ref": target.text})

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        self.val_predictions = []

    def on_test_epoch_start(self) -> None:
        self.test_predictions = []

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )
