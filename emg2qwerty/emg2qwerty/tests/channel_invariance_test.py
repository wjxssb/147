"""Tests for electrode channel masking and excluded-channel invariance.

Verifies that:
1. MaskChannels zeroes out unselected electrodes/bands before STFT.
2. Invariance property: Modifying excluded channels in the raw sEMG signal has
   EXACTLY ZERO effect on the model frontend features and decoded logits.
3. Bug regression check: Placing RandomBandRotation before MaskChannels leaks
   excluded physical electrodes into active channels (proven via synthetic identities).
4. SpectrogramNorm normalizes channels independently without cross-channel leakage.
"""

import numpy as np
import pytest
import torch

from emg2qwerty.lightning import CNNBiGRUCTCModule
from emg2qwerty.modules import SpectrogramNorm
from emg2qwerty.transforms import (
    ForEach,
    LogSpectrogram,
    MaskChannels,
    RandomBandRotation,
    TemporalAlignmentJitter,
)


def test_synthetic_channel_identity_leakage_demonstration():
    """Demonstrates that rolling before masking contaminates active slots with excluded electrodes."""
    # Synthetic tensor: each channel c has value float(c)
    x = torch.zeros(10, 2, 16)
    for c in range(16):
        x[:, :, c] = float(c)

    # Buggy order: rotation then mask
    rot = RandomBandRotation(offsets=[1])
    mask = MaskChannels(keep_electrodes=[0, 2, 4, 6, 8, 10, 12, 14])

    x_rot = rot(x)
    x_masked = mask(x_rot)

    active_indices = [0, 2, 4, 6, 8, 10, 12, 14]
    active_values = [int(v) for v in x_masked[0, 0, active_indices].tolist()]

    # When shifted by +1, active positions now contain physical channels [15, 1, 3, 5, 7, 9, 11, 13]
    assert active_values == [15, 1, 3, 5, 7, 9, 11, 13], "Expected odd channels rolled into active slots!"
    leaked_odd = [v for v in active_values if v % 2 != 0]
    assert len(leaked_odd) == 8, "All active slots were contaminated by excluded odd electrodes!"


def test_clean_pipeline_channel_identity():
    """Verifies that in the clean pipeline (rotation disabled or after mask), only even channels survive."""
    x = torch.zeros(10, 2, 16)
    for c in range(16):
        x[:, :, c] = float(c)

    # Clean order: mask channels directly, no rotation
    mask = MaskChannels(keep_electrodes=[0, 2, 4, 6, 8, 10, 12, 14])
    x_masked = mask(x)

    # Even channels must retain their exact original values
    for c in [0, 2, 4, 6, 8, 10, 12, 14]:
        assert torch.all(x_masked[:, :, c] == float(c))

    # Odd (excluded) channels must be strictly 0.0
    for c in [1, 3, 5, 7, 9, 11, 13, 15]:
        assert torch.all(x_masked[:, :, c] == 0.0)


def test_spectrogram_norm_channel_independence():
    """Confirms 2D BatchNorm over (N, freq, T) processes each channel independently."""
    torch.manual_seed(42)
    norm = SpectrogramNorm(channels=32)
    norm.eval()

    # Input: (T=50, N=2, bands=2, C=16, freq=33)
    x1 = torch.randn(50, 2, 2, 16, 33)
    x2 = x1.clone()

    # Mask odd channels in band 0 to zero in x1
    x1[:, :, 0, 1::2, :] = 0.0
    # In x2, put arbitrary high-amplitude values in odd channels
    x2[:, :, 0, 1::2, :] = torch.randn(50, 2, 8, 33) * 50.0 + 100.0

    y1 = norm(x1)
    y2 = norm(x2)

    # Active (even) channels in band 0 must be 100% IDENTICAL
    diff_even = (y1[:, :, 0, 0::2, :] - y2[:, :, 0, 0::2, :]).abs().max().item()
    assert diff_even == 0.0, f"SpectrogramNorm leaked between channels! diff={diff_even}"

    # Band 1 channels must be 100% IDENTICAL
    diff_band1 = (y1[:, :, 1, :, :] - y2[:, :, 1, :, :]).abs().max().item()
    assert diff_band1 == 0.0, f"SpectrogramNorm leaked across bands! diff={diff_band1}"


@pytest.mark.parametrize("keep_config, corrupt_dim, corrupt_indices", [
    # F3: Dual 8+8 (keep even electrodes, corrupt odd electrodes)
    ({"keep_electrodes": [0, 2, 4, 6, 8, 10, 12, 14]}, "electrodes", [1, 3, 5, 7, 9, 11, 13, 15]),
    # F1: Left wrist only (keep band 0, corrupt band 1)
    ({"keep_bands": [0]}, "bands", [1]),
    # F2: Right wrist only (keep band 1, corrupt band 0)
    ({"keep_bands": [1]}, "bands", [0]),
])
def test_excluded_channel_invariance_end_to_end(keep_config, corrupt_dim, corrupt_indices):
    """End-to-end excluded-channel invariance test.

    Injecting arbitrary extreme noise into the excluded physical channels
    MUST produce 0.0 change in model output logits.
    """
    torch.manual_seed(42)
    module = CNNBiGRUCTCModule(
        in_features=528,
        mlp_features=[384],
        hidden_size=256,
        num_layers=2,
        num_bands=2,
        electrode_channels=16,
    )
    module.eval()

    # Raw sEMG input: (T=8000, bands=2, channels=16)
    raw1 = torch.randn(8000, 2, 16)
    raw2 = raw1.clone()

    # Corrupt the excluded channels with massive random values
    if corrupt_dim == "electrodes":
        raw2[:, :, corrupt_indices] = torch.randn(8000, 2, len(corrupt_indices)) * 1000.0 - 500.0
    elif corrupt_dim == "bands":
        raw2[:, corrupt_indices, :] = torch.randn(8000, len(corrupt_indices), 16) * 1000.0 - 500.0

    mask = MaskChannels(**keep_config)
    logspec = LogSpectrogram(n_fft=64, hop_length=16)

    # Clean transform: mask first, then logspec
    t1 = logspec(mask(raw1)).unsqueeze(1)  # (T_spec, N=1, bands=2, C=16, freq=33)
    t2 = logspec(mask(raw2)).unsqueeze(1)

    spec_diff = (t1 - t2).abs().max().item()
    assert spec_diff == 0.0, f"Spectrogram difference detected: {spec_diff}"

    with torch.no_grad():
        lengths = torch.tensor([t1.shape[0]])
        logits1 = module(t1, lengths)
        logits2 = module(t2, lengths)

    logits_diff = (logits1 - logits2).abs().max().item()
    assert logits_diff == 0.0, f"Model prediction changed when modifying excluded channels! diff={logits_diff}"
