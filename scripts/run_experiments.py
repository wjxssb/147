#!/usr/bin/env python3
"""
scripts/run_experiments.py
===========================
Automated experimental harness for UCLA ECE C147A/C247A Final Project.
Executes training and evaluation across the official experimental matrix:
  - A0: TDS-CNN Baseline (32ch, Dual-wrist)
  - A1: CNN-BiGRU Main Model (32ch, Dual-wrist)
  - C1: Single Left Wrist (16L, 16ch)
  - C2: Single Right Wrist (16R, 16ch)
  - C3: Equidistant Dual Wrist (8L + 8R, 16ch)
  - C4: Sparse Dual Wrist (4L + 4R, 8ch)

Features:
- Fixed seeds (default: 42, then 123, 2026).
- Strict separation of validation and test evaluation (no test leakage).
- Hardware profiling: Peak VRAM, epoch duration, parameter count.
- Structured output logging: checkpoints, metrics.json, hardware_metrics.json.
- Graceful error recovery and resumable state.
"""

import os
import sys
import time
import json
import argparse
import subprocess
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Channel Ablation Configurations
EXPERIMENT_CONFIGS = {
    "A0_TDS_Baseline": {
        "model": "tds_conv_ctc",
        "description": "Official TDS-CNN Baseline (32 channels, 2 bands x 16 channels)",
        "overrides": [
            "model=tds_conv_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
        ],
    },
    "A1_CNN_BiGRU": {
        "model": "cnn_bigru_ctc",
        "description": "CNN-BiGRU Main Architecture (32 channels, 2 bands x 16 channels)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
        ],
    },
    "A1_CNN_BiGRU_Clean": {
        "model": "cnn_bigru_ctc",
        "description": "Clean Full 32ch CNN-BiGRU (Rotation Disabled, Identical Clean Protocol to F3-Clean)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "band_rotation.transform.offsets=[0]",
        ],
    },
    "C1_Left_16ch": {
        "model": "cnn_bigru_ctc",
        "description": "Left wrist only (1 band x 16 channels, C1 ablation)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=1",
            "module.electrode_channels=16",
            "to_tensor.fields=['emg_left']",
        ],
    },
    "C2_Right_16ch": {
        "model": "cnn_bigru_ctc",
        "description": "Right wrist only (1 band x 16 channels, C2 ablation)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=1",
            "module.electrode_channels=16",
            "to_tensor.fields=['emg_right']",
        ],
    },
    "C3_Dual_8ch": {
        "model": "cnn_bigru_ctc",
        "description": "Equidistant dual-wrist (8L + 8R = 16 channels total)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=264",  # 8 channels * 33 freq bins = 264
            "module.num_bands=2",
            "module.electrode_channels=8",
            "to_tensor.indices=[0,2,4,6,8,10,12,14]",
        ],
    },
    "C4_Dual_4ch": {
        "model": "cnn_bigru_ctc",
        "description": "Sparse dual-wrist (4L + 4R = 8 channels total)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=132",  # 4 channels * 33 freq bins = 132
            "module.num_bands=2",
            "module.electrode_channels=4",
            "to_tensor.indices=[0,4,8,12]",
        ],
    },
    "F1_Left_16ch_FixedCap": {
        "model": "cnn_bigru_ctc",
        "description": "Fixed-capacity single left wrist (16L, 16ch, fixed 5.4M backbone via channel masking)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[0]",
        ],
    },
    "F2_Right_16ch_FixedCap": {
        "model": "cnn_bigru_ctc",
        "description": "Fixed-capacity single right wrist (16R, 16ch, fixed 5.4M backbone via channel masking)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[1]",
        ],
    },
    "F3_Dual_8ch_FixedCap": {
        "model": "cnn_bigru_ctc",
        "description": "Legacy Fixed-capacity equidistant dual wrist (8L + 8R = 16ch, fixed 5.4M backbone, contaminated rotation order)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_electrodes=[0,2,4,6,8,10,12,14]",
        ],
    },
    "F1_Left_16ch_Clean": {
        "model": "cnn_bigru_ctc",
        "description": "Clean Fixed-capacity Left Wrist (16L, 5.4M, Mask First, Rotation Disabled)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[0]",
            "band_rotation.transform.offsets=[0]",
        ],
    },
    "F2_Right_16ch_Clean": {
        "model": "cnn_bigru_ctc",
        "description": "Clean Fixed-capacity Right Wrist (16R, 5.4M, Mask First, Rotation Disabled)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[1]",
            "band_rotation.transform.offsets=[0]",
        ],
    },
    "F3_Dual_8ch_Clean": {
        "model": "cnn_bigru_ctc",
        "description": "Clean Fixed-capacity Dual 8+8 (16ch, 5.4M, Mask First, Rotation Disabled)",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_electrodes=[0,2,4,6,8,10,12,14]",
            "band_rotation.transform.offsets=[0]",
        ],
    },
}

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_single_experiment(
    exp_id: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
    evaluate_test: bool = False,
    results_dir: Path | None = None,
):
    if exp_id not in EXPERIMENT_CONFIGS:
        raise ValueError(f"Unknown experiment ID: {exp_id}. Available: {list(EXPERIMENT_CONFIGS.keys())}")

    exp_info = EXPERIMENT_CONFIGS[exp_id]
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / f"{exp_id}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {exp_id} (Seed: {seed})")
    print(f"Description: {exp_info['description']}")
    print(f"Output Directory: {results_dir}")
    print(f"{'='*70}\n")

    # Construct Hydra command
    cmd = [
        sys.executable,
        "-m", "emg2qwerty.train",
        f"user=single_user",
        f"seed={seed}",
        f"batch_size={batch_size}",
        f"trainer.max_epochs={epochs}",
        f"trainer.accelerator={'gpu' if 'cuda' in device or device == 'gpu' else 'cpu'}",
        f"trainer.devices=1",
        f"trainer.gradient_clip_val=1.0",
        f"hydra.run.dir={results_dir}",
        f"evaluate_test={str(evaluate_test).lower()}",
    ]
    cmd.extend(exp_info["overrides"])

    # Track start time and initial GPU memory
    start_time = time.time()
    initial_vram = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0

    print("Command line:")
    print(" ".join(cmd))
    print()

    # Execute training
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    process = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))

    elapsed_time = time.time() - start_time
    peak_vram = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0

    if process.returncode != 0:
        print(f"\n[ERROR] Experiment {exp_id} failed with exit code {process.returncode}")
        return False

    # Check for results.json saved by train.py
    results_file = results_dir / "results.json"
    child_peak_vram = 0.0
    if results_file.exists():
        try:
            with open(results_file) as f:
                res_data = json.load(f)
                child_peak_vram = res_data.get("peak_vram_mb", 0.0)
        except Exception:
            pass

    # Save hardware and execution metrics
    hardware_info = {
        "experiment_id": exp_id,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": device,
        "training_time_seconds": elapsed_time,
        "training_time_minutes": elapsed_time / 60.0,
        "peak_vram_mb": child_peak_vram if child_peak_vram > 0 else (peak_vram / (1024 * 1024)),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(results_dir / "hardware_metrics.json", "w") as f:
        json.dump(hardware_info, f, indent=2)

    # Automatic error analysis on validation predictions
    val_pred_file = results_dir / "val_predictions.json"
    if val_pred_file.exists():
        err_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "error_analysis.py"),
            "--pred_file", str(val_pred_file),
            "--out_file", str(results_dir / "error_analysis.json"),
        ]
        subprocess.run(err_cmd, env=env, cwd=str(PROJECT_ROOT))

    print(f"\n✓ Experiment {exp_id} (Seed {seed}) completed in {elapsed_time/60.0:.2f} mins.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run ECE C147A Experiments")
    parser.add_argument("--experiments", nargs="+", default=["A0_TDS_Baseline", "A1_CNN_BiGRU"],
                        help="List of experiment IDs to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="List of random seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16 for VRAM safety)")
    parser.add_argument("--device", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate_test", action="store_true", help="Evaluate frozen test set after training")
    args = parser.parse_args()

    for exp_id in args.experiments:
        for seed in args.seeds:
            success = run_single_experiment(
                exp_id=exp_id,
                seed=seed,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=args.device,
                evaluate_test=args.evaluate_test,
            )
            if not success:
                print(f"Stopping execution due to error in {exp_id}")
                sys.exit(1)

if __name__ == "__main__":
    main()
