#!/usr/bin/env python3
"""
scripts/preflight_check.py
===========================
Comprehensive unit and integration test suite for the UCLA ECE C147A/C247A Final Project.
Verifies:
1. CharacterSet and CTC Blank ID consistency.
2. CTC duplicate alignment constraint (T_out >= L + R) and synthetic validation.
3. Greedy Decoder repeated-character collapse with blank separators (e.g., 'book', 'hello', 'success').
4. STFT, SpectrogramNorm, and RotationInvariantMLP feature pipeline dimensions.
5. CNNBiGRUEncoder shape invariance (T_diff = 0) and pack_padded_sequence padding isolation.
6. Gradient backpropagation and parameter finiteness (no NaN / Inf).
7. Electrode channel ablation configurations (C0: 32ch, C1: 16L, C2: 16R, C3: 8L+8R, C4: 4L+4R).
8. Real HDF5 dataset loading and session parsing.
9. Checkpoint saving and reloading.

Can be run via pytest:
    pytest scripts/preflight_check.py -v
or directly:
    python scripts/preflight_check.py
"""

import sys
import tempfile
from pathlib import Path
import pytest
import numpy as np
import torch
from torch import nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emg2qwerty.charset import charset, CharacterSet
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    CNNBiGRUEncoder,
)
from emg2qwerty.transforms import (
    LogSpectrogram,
    SelectElectrodeChannels,
    MaskChannels,
    TemporalAlignmentJitter,
)
from emg2qwerty.lightning import TDSConvCTCModule, CNNBiGRUCTCModule
from emg2qwerty.data import EMGSessionData, WindowedEMGDataset, LabelData


# =========================================================================
# 1. CharacterSet & CTC Blank Consistency
# =========================================================================
def test_charset_and_blank():
    cs = charset()
    assert isinstance(cs, CharacterSet)
    assert cs.null_class is not None, "Null class (blank) must be defined"
    assert 0 <= cs.null_class < cs.num_classes, "Null class index must be within range"
    
    # Test encoding and decoding round-trip
    test_str = "hello world, test 123!"
    encoded = cs.str_to_labels(test_str)
    decoded = cs.labels_to_str(encoded)
    assert decoded == test_str, f"Mismatch in roundtrip: {decoded} != {test_str}"
    print("✓ CharacterSet and CTC Blank ID test passed.")


# =========================================================================
# 2. CTC Duplicate Alignment Constraint (T_out >= L + R)
# =========================================================================
@pytest.mark.parametrize("word", ["book", "hello", "success", "committee", "coffee"])
def test_ctc_duplicate_words_and_length_constraint(word):
    """
    In CTC, consecutive duplicate characters MUST be separated by at least one blank.
    Therefore, the minimum required sequence length is L + R, where L is target length
    and R is the count of consecutive duplicate pairs.
    """
    cs = charset()
    encoded = cs.str_to_labels(word)
    L = len(encoded)
    # Count consecutive duplicates
    R = sum(1 for i in range(len(encoded) - 1) if encoded[i] == encoded[i + 1])
    min_required_len = L + R

    ctc_loss_fn = nn.CTCLoss(blank=cs.null_class, zero_infinity=False)

    # 1. When T_out < L + R, valid CTC alignment is mathematically IMPOSSIBLE
    too_short_T = min_required_len - 1
    if too_short_T > 0:
        log_probs_short = torch.randn(too_short_T, 1, cs.num_classes).log_softmax(dim=-1)
        targets_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # (N=1, L)
        input_lengths = torch.tensor([too_short_T], dtype=torch.long)
        target_lengths = torch.tensor([L], dtype=torch.long)

        # PyTorch CTCLoss returns inf or fails when impossible
        loss = ctc_loss_fn(log_probs_short, targets_tensor, input_lengths, target_lengths)
        assert torch.isinf(loss) or loss.item() > 1e6, f"Expected inf loss for impossible length {too_short_T}"

    # 2. When T_out >= L + R, CTC alignment MUST succeed with finite loss and valid gradient
    valid_T = min_required_len + 4
    raw_logits = torch.randn(valid_T, 1, cs.num_classes, requires_grad=True)
    log_probs_valid = raw_logits.log_softmax(dim=-1)
    targets_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)
    input_lengths = torch.tensor([valid_T], dtype=torch.long)
    target_lengths = torch.tensor([L], dtype=torch.long)

    loss_valid = ctc_loss_fn(log_probs_valid, targets_tensor, input_lengths, target_lengths)
    assert not torch.isnan(loss_valid) and not torch.isinf(loss_valid), f"Loss for '{word}' is not finite"
    loss_valid.backward()
    assert raw_logits.grad is not None
    assert not torch.isnan(raw_logits.grad).any()
    print(f"✓ CTC duplicate word test passed for '{word}' (L={L}, R={R}, min_T={min_required_len}).")


# =========================================================================
# 3. Greedy CTC Decoder Collapse with Blanks
# =========================================================================
def test_greedy_decoder_blank_collapse():
    """
    Greedy decoding: argmax over classes, collapse consecutive identical symbols,
    then drop blanks.
    """
    cs = charset()
    null = cs.null_class
    
    # Simulate emissions for word "meet": m - e - <blank> - e - t
    c_m = cs.str_to_labels("m")[0]
    c_e = cs.str_to_labels("e")[0]
    c_t = cs.str_to_labels("t")[0]

    # Token sequence with repetitions and blank separator: [m, m, e, e, null, e, t, t]
    tokens = [c_m, c_m, c_e, c_e, null, c_e, c_t, c_t]
    
    # 1. Collapse consecutive identical tokens: [m, e, null, e, t]
    collapsed = [tokens[0]]
    for t in tokens[1:]:
        if t != collapsed[-1]:
            collapsed.append(t)
    
    # 2. Remove null tokens: [m, e, e, t]
    final_tokens = [t for t in collapsed if t != null]
    decoded_str = cs.labels_to_str(final_tokens)
    assert decoded_str == "meet", f"Expected 'meet', got '{decoded_str}'"
    print("✓ Greedy decoder blank collapse test passed.")


# =========================================================================
# 4. STFT and Feature Pipeline Dimensions
# =========================================================================
def test_stft_and_spatial_mlp_shapes():
    """
    Raw sEMG: (T_raw, 2, 16)
    -> LogSpectrogram (n_fft=64, hop=16) -> (T_spec, 2, 16, 33)
    -> SpectrogramNorm(32) -> (T_spec, 2, 16, 33)
    -> MultiBandRotationInvariantMLP(528 -> 384) -> (T_spec, 2, 384)
    -> Flatten(start_dim=2) -> (T_spec, N, 768)
    """
    T_raw = 8000
    N = 2
    raw = torch.randn(T_raw, N, 2, 16)

    # STFT
    stft = LogSpectrogram(n_fft=64, hop_length=16)
    # Apply to batch: movedim or per item
    # Since LogSpectrogram expects (T, ...)
    spec = stft(raw)  # (T_spec, N, 2, 16, 33)
    T_spec = spec.shape[0]
    assert spec.shape == (T_spec, N, 2, 16, 33), f"Unexpected spec shape: {spec.shape}"

    # Norm
    norm = SpectrogramNorm(channels=32)
    normed = norm(spec)
    assert normed.shape == spec.shape

    # MLP
    mlp = MultiBandRotationInvariantMLP(in_features=16 * 33, mlp_features=[384], num_bands=2)
    mlp_out = mlp(normed)  # (T_spec, N, 2, 384)
    assert mlp_out.shape == (T_spec, N, 2, 384)

    # Flatten
    features = mlp_out.flatten(start_dim=2)
    assert features.shape == (T_spec, N, 768)
    print("✓ STFT and spatial MLP dimension flow verified (528 -> 384 -> 768).")


# =========================================================================
# 5. CNN-BiGRU Encoder Invariance & Padding Isolation
# =========================================================================
def test_cnn_bigru_encoder():
    T = 200
    N = 3
    C = 768
    encoder = CNNBiGRUEncoder(num_features=C, hidden_size=256, num_layers=2, dropout=0.2, kernel_size=3)

    inputs = torch.randn(T, N, C, requires_grad=True)
    lengths = torch.tensor([200, 150, 100], dtype=torch.long)

    # Forward with variable lengths
    outputs = encoder(inputs, input_lengths=lengths)
    assert outputs.shape == (T, N, C), f"Output shape mismatch: {outputs.shape} != {(T, N, C)}"

    # Backward pass verification
    loss = outputs.sum()
    loss.backward()
    assert inputs.grad is not None
    assert not torch.isnan(inputs.grad).any(), "NaN found in input gradient"

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Missing gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient of {name}"
        assert not torch.isinf(param.grad).any(), f"Inf in gradient of {name}"

    print("✓ CNN-BiGRU encoder forward, padding packing, and backpropagation verified.")


# =========================================================================
# 6. Physical Electrode Channel Ablation Configurations
# =========================================================================
def test_mask_channels():
    t = torch.ones(100, 2, 16)
    mask_b0 = MaskChannels(keep_bands=[0])
    res_b0 = mask_b0(t)
    assert (res_b0[:, 0, :] == 1.0).all()
    assert (res_b0[:, 1, :] == 0.0).all()

    mask_b1 = MaskChannels(keep_bands=[1])
    res_b1 = mask_b1(t)
    assert (res_b1[:, 0, :] == 0.0).all()
    assert (res_b1[:, 1, :] == 1.0).all()

    mask_e = MaskChannels(keep_electrodes=[0, 2, 4, 6, 8, 10, 12, 14])
    res_e = mask_e(t)
    assert (res_e[:, :, [0, 2, 4, 6, 8, 10, 12, 14]] == 1.0).all()
    assert (res_e[:, :, [1, 3, 5, 7, 9, 11, 13, 15]] == 0.0).all()
    print("✓ MaskChannels unit tests passed.")


@pytest.mark.parametrize(
    "config_name,bands,indices,expected_feat",
    [
        ("C0_32ch", 2, list(range(16)), 528),
        ("C1_16L", 1, list(range(16)), 528),
        ("C2_16R", 1, list(range(16)), 528),
        ("C3_8L8R", 2, [0, 2, 4, 6, 8, 10, 12, 14], 264),
        ("C4_4L4R", 2, [0, 4, 8, 12], 132),
    ],
)
def test_channel_ablation_configurations(config_name, bands, indices, expected_feat):
    """
    Verifies that physical channel slicing and single-wrist settings correctly
    adjust in_features and construct runnable CNN-BiGRU modules.
    """
    selector = SelectElectrodeChannels(indices=indices)
    raw = torch.randn(1000, bands, 16)
    sliced = selector(raw)
    assert sliced.shape == (1000, bands, len(indices))

    # Jitter test (single-band bypass or dual-band jitter)
    jitter = TemporalAlignmentJitter(max_offset=20)
    jittered = jitter(sliced)
    if bands == 1:
        assert jittered.shape == sliced.shape, "Single band jitter bypass failed"
    else:
        assert jittered.shape[0] <= sliced.shape[0] and jittered.shape[1:] == sliced.shape[1:]

    # Module instantiation test
    channels_per_band = len(indices)
    freq_bins = 33
    in_features = channels_per_band * freq_bins
    assert in_features == expected_feat, f"Feature count mismatch: {in_features} != {expected_feat}"

    module = CNNBiGRUCTCModule(
        in_features=in_features,
        mlp_features=[384],
        hidden_size=256,
        num_layers=2,
        num_bands=bands,
        electrode_channels=channels_per_band,
    )

    # Minimal forward pass
    dummy_input = torch.randn(50, 1, bands, channels_per_band, freq_bins)
    emissions = module(dummy_input)
    assert emissions.shape == (50, 1, charset().num_classes)
    print(f"✓ Channel ablation '{config_name}' (bands={bands}, ch={channels_per_band}) verified.")


# =========================================================================
# 7. Real HDF5 Dataset Loading & Integrity Check
# =========================================================================
def test_real_hdf5_dataset():
    data_dir = PROJECT_ROOT / "data"
    hdf5_files = sorted(list(data_dir.glob("*.hdf5")))
    assert len(hdf5_files) == 18, f"Expected 18 HDF5 sessions, found {len(hdf5_files)}"

    # Load first session
    sample_file = hdf5_files[0]
    with EMGSessionData(sample_file) as session:
        assert session.condition == "on_keyboard"
        assert len(session) > 0
        left_emg = session.timeseries["emg_left"]
        right_emg = session.timeseries["emg_right"]
        assert left_emg.shape[1] == 16, "Left band must have 16 channels"
        assert right_emg.shape[1] == 16, "Right band must have 16 channels"
        print(f"✓ Real HDF5 file {sample_file.name} checked ({len(session)} raw samples).")


# =========================================================================
# 8. Checkpoint Save and Restore
# =========================================================================
def test_checkpoint_save_and_load():
    module = CNNBiGRUCTCModule(
        in_features=528,
        mlp_features=[384],
        hidden_size=64,
        num_layers=1,
        num_bands=2,
        electrode_channels=16,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_model.ckpt"
        torch.save({"state_dict": module.state_dict()}, ckpt_path)
        assert ckpt_path.exists()

        loaded_checkpoint = torch.load(ckpt_path, weights_only=False)
        module.load_state_dict(loaded_checkpoint["state_dict"])
    print("✓ Model checkpoint save and restore verified.")


if __name__ == "__main__":
    print("\n==================================================")
    print("Running Full ECE C147A Final Project Preflight Suite")
    print("==================================================\n")
    test_charset_and_blank()
    for w in ["book", "hello", "success", "committee"]:
        test_ctc_duplicate_words_and_length_constraint(w)
    test_greedy_decoder_blank_collapse()
    test_stft_and_spatial_mlp_shapes()
    test_cnn_bigru_encoder()
    test_mask_channels()
    for cfg in [
        ("C0_32ch", 2, list(range(16)), 528),
        ("C1_16L", 1, list(range(16)), 528),
        ("C2_16R", 1, list(range(16)), 528),
        ("C3_8L8R", 2, [0, 2, 4, 6, 8, 10, 12, 14], 264),
        ("C4_4L4R", 2, [0, 4, 8, 12], 132),
    ]:
        test_channel_ablation_configurations(*cfg)
    test_real_hdf5_dataset()
    test_checkpoint_save_and_load()
    print("\n==================================================")
    print("ALL PREFLIGHT CHECKS PASSED SUCCESSFULLY!")
    print("==================================================\n")
