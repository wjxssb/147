#!/usr/bin/env python3
"""
scripts/evaluate_frozen_test.py
================================
Evaluates frozen checkpoints on the official test set (Session 2021-06-02-1622682789).
Maintains strict separation between validation tuning and test evaluation.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPERIMENTS = [
    {
        "name": "A0_TDS_Baseline_seed42",
        "model": "tds_conv_ctc",
        "overrides": [
            "model=tds_conv_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
        ],
    },
    {
        "name": "A1_CNN_BiGRU_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
        ],
    },
    {
        "name": "C1_Left_16ch_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=1",
            "module.electrode_channels=16",
            "to_tensor.fields=['emg_left']",
        ],
    },
    {
        "name": "C2_Right_16ch_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=1",
            "module.electrode_channels=16",
            "to_tensor.fields=['emg_right']",
        ],
    },
    {
        "name": "C3_Dual_8ch_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=264",
            "module.num_bands=2",
            "module.electrode_channels=8",
            "to_tensor.indices=[0,2,4,6,8,10,12,14]",
        ],
    },
    {
        "name": "C4_Dual_4ch_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=132",
            "module.num_bands=2",
            "module.electrode_channels=4",
            "to_tensor.indices=[0,4,8,12]",
        ],
    },
    {
        "name": "F1_Left_16ch_FixedCap_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[0]",
        ],
    },
    {
        "name": "F2_Right_16ch_FixedCap_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_bands=[1]",
        ],
    },
    {
        "name": "F3_Dual_8ch_FixedCap_seed42",
        "model": "cnn_bigru_ctc",
        "overrides": [
            "model=cnn_bigru_ctc",
            "module.in_features=528",
            "module.num_bands=2",
            "module.electrode_channels=16",
            "channel_mask.keep_electrodes=[0,2,4,6,8,10,12,14]",
        ],
    },
]

def find_checkpoint(exp_dir: Path) -> Path:
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    # Prioritize epoch checkpoint over last.ckpt
    ckpts = [p for p in ckpt_dir.glob("epoch=*.ckpt")]
    if ckpts:
        return ckpts[0]
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

def evaluate_test():
    results_root = PROJECT_ROOT / "results"
    summary = {}

    for exp in EXPERIMENTS:
        exp_name = exp["name"]
        exp_dir = results_root / exp_name
        if not exp_dir.exists():
            print(f"Skipping {exp_name}: folder does not exist.")
            continue

        ckpt_path = find_checkpoint(exp_dir)
        print(f"\n{'='*70}")
        print(f"EVALUATING TEST SET FOR: {exp_name}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable,
            "-m", "emg2qwerty.train",
            "user=single_user",
            "train=False",
            "evaluate_test=True",
            f"checkpoint='{ckpt_path}'",
            "trainer.accelerator=gpu",
            "trainer.devices=1",
            f"hydra.run.dir={exp_dir}",
            f"trainer.default_root_dir={exp_dir}",
        ]
        cmd.extend(exp["overrides"])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        ret = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        if ret.returncode != 0:
            print(f"[ERROR] Failed to evaluate test set for {exp_name}")
            continue

        # Run error analysis on test predictions if available
        test_pred_file = exp_dir / "test_predictions.json"
        if test_pred_file.exists():
            err_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "error_analysis.py"),
                "--pred_file", str(test_pred_file),
                "--out_file", str(exp_dir / "test_error_analysis.json"),
            ]
            subprocess.run(err_cmd, env=env, cwd=str(PROJECT_ROOT))

        # Check results.json
        res_file = exp_dir / "results.json"
        if res_file.exists():
            with open(res_file) as f:
                res = json.load(f)
                test_m = res.get("test_metrics")
                if test_m and isinstance(test_m, list) and len(test_m) > 0:
                    summary[exp_name] = test_m[0]

    print("\n" + "="*70)
    print("FROZEN TEST EVALUATION COMPLETE. SUMMARY:")
    print("="*70)
    for k, v in summary.items():
        print(f"{k}: CER = {v.get('test/CER', 'N/A')}% | Loss = {v.get('test/loss', 'N/A')}")

    with open(results_root / "test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    evaluate_test()
