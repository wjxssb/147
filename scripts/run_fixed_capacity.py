#!/usr/bin/env python3
"""
scripts/run_fixed_capacity.py
=============================
Trains and evaluates the fixed-capacity masked-channel control models:
  - F1_Left_16ch_FixedCap (seed 42, 5.4M params)
  - F2_Right_16ch_FixedCap (seed 42, 5.4M params)
  - F3_Dual_8ch_FixedCap (seed 42, 5.4M params)

Matches the exact 5,422,944 parameter capacity of A1 CNN-BiGRU (Dual 16+16),
eliminating parameter capacity as a confounding factor in electrode ablation.
"""

import sys
import subprocess
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    experiments = [
        "F1_Left_16ch_FixedCap",
        "F2_Right_16ch_FixedCap",
        "F3_Dual_8ch_FixedCap",
    ]
    
    print("\n=======================================================")
    print("STEP 1: TRAINING FIXED-CAPACITY ABLATIONS (F1, F2, F3)")
    print("=======================================================\n")
    
    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_experiments.py"),
        "--experiments", *experiments,
        "--seeds", "42",
        "--epochs", "35",
        "--batch_size", "16",
    ]
    
    res = subprocess.run(train_cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print("[ERROR] Training failed!")
        sys.exit(res.returncode)
        
    print("\n=======================================================")
    print("STEP 2: EVALUATING FROZEN TEST SET ON F1, F2, F3")
    print("=======================================================\n")
    
    test_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_frozen_test.py"),
    ]
    res = subprocess.run(test_cmd, cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        print("[ERROR] Test evaluation failed!")
        sys.exit(res.returncode)
        
    print("\n=======================================================")
    print("ALL FIXED-CAPACITY EXPERIMENTS COMPLETED!")
    print("=======================================================\n")

if __name__ == "__main__":
    main()
