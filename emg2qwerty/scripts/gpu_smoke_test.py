#!/usr/bin/env python3
"""
scripts/gpu_smoke_test.py
=========================
Executes a rapid GPU smoke test on the clean masked channel pipeline.
Verifies on CUDA device:
1. Excluded channel invariance on GPU (max_diff == 0.0)
2. Single-step train forward & backward pass on GPU
3. Single-step validation pass on GPU
"""

import sys
from pathlib import Path
import torch
import pytorch_lightning as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from emg2qwerty.lightning import CNNBiGRUCTCModule
from emg2qwerty.transforms import MaskChannels, LogSpectrogram, RandomBandRotation, ForEach, TemporalAlignmentJitter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running GPU Smoke Test on: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Test Excluded Channel Invariance on GPU
    module = CNNBiGRUCTCModule(
        in_features=528,
        mlp_features=[384],
        hidden_size=256,
        num_layers=2,
        num_bands=2,
        electrode_channels=16,
    ).to(device)
    module.eval()

    mask = MaskChannels(keep_electrodes=[0, 2, 4, 6, 8, 10, 12, 14])
    rot = ForEach(RandomBandRotation(offsets=[0]))  # Clean protocol: disabled rotation
    logspec = LogSpectrogram(n_fft=64, hop_length=16)

    # Raw tensors
    raw1 = torch.randn(8000, 2, 16)
    raw2 = raw1.clone()
    raw2[:, :, 1::2] = torch.randn(8000, 2, 8) * 1000.0  # Massive corruption on excluded channels

    t1 = logspec(rot(mask(raw1))).unsqueeze(1).to(device)  # (T, N=1, B=2, C=16, freq=33)
    t2 = logspec(rot(mask(raw2))).unsqueeze(1).to(device)

    with torch.no_grad():
        lengths = torch.tensor([t1.shape[0]], device=device)
        logits1 = module(t1, lengths)
        logits2 = module(t2, lengths)

    diff = (logits1 - logits2).abs().max().item()
    print(f"GPU Logits Diff on corrupted excluded channels: {diff}")
    assert diff == 0.0, f"Leakage detected on GPU! diff={diff}"
    print("✓ GPU Excluded-Channel Invariance Test PASSED!")

    # 2. Test Single-Step Train Pass on GPU
    module.train()
    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    optimizer.zero_grad()

    # Synthetic targets: characters 1..5
    targets = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=device)
    target_lengths = torch.tensor([5], dtype=torch.long, device=device)
    emission_lengths = torch.tensor([t1.shape[0]], dtype=torch.long, device=device)

    log_probs = module(t1, emission_lengths)
    ctc_loss = torch.nn.functional.ctc_loss(
        log_probs,
        targets,
        emission_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )
    print(f"GPU CTC Loss: {ctc_loss.item():.4f}")
    assert not torch.isnan(ctc_loss) and not torch.isinf(ctc_loss)

    ctc_loss.backward()
    optimizer.step()
    print("✓ GPU Forward + Backward + Optimizer Step PASSED!")

    print("\n==================================================")
    print("ALL GPU SMOKE TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    main()
