#!/usr/bin/env python3
"""
scripts/aggregate_results.py
=============================
Aggregates experimental outputs across configurations and random seeds.
Generates:
1. Markdown summary comparison tables.
2. NeurIPS 2024 LaTeX formatted tables for the paper write-up.
3. Statistical Mean ± Std across seeds for each configuration.
4. Error rate breakdowns (CER, Substitution, Deletion, Insertion).
5. Hardware efficiency metrics (Parameters, Training Time, VRAM Peak, Inference Latency).
"""

import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_metrics(results_dir: Path):
    """
    Scans results_dir for experiment folders and extracts metrics.
    """
    records = []
    for exp_folder in results_dir.glob("*_seed*"):
        if not exp_folder.is_dir():
            continue
        
        # Parse name: <exp_id>_seed<seed>
        parts = exp_folder.name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        exp_id, seed_str = parts[0], parts[1]
        seed = int(seed_str)

        # Load hardware metrics
        hw_file = exp_folder / "hardware_metrics.json"
        hw_data = {}
        if hw_file.exists():
            with open(hw_file) as f:
                hw_data = json.load(f)

        # Load CER / evaluation metrics from results.json and error_analysis.json
        eval_data = {}
        res_file = exp_folder / "results.json"
        if res_file.exists():
            try:
                with open(res_file) as f:
                    r_raw = json.load(f)
                    val_m = r_raw.get("val_metrics", [{}])[0]
                    # CER in Lightning metrics is percentage (e.g. 30.5 for 30.5%)
                    raw_cer = val_m.get("val/CER", None)
                    if raw_cer is not None:
                        eval_data["val/CER"] = raw_cer / 100.0 if raw_cer > 1.0 else raw_cer
                    eval_data["val/loss"] = val_m.get("val/loss", None)
            except Exception:
                pass

        err_file = exp_folder / "error_analysis.json"
        if err_file.exists():
            try:
                with open(err_file) as f:
                    err_raw = json.load(f)
                    overall = err_raw.get("overall", {})
                    eval_data["val/CER"] = overall.get("CER", eval_data.get("val/CER", 0.0))
                    eval_data["val/sub_rate"] = overall.get("sub_rate", 0.0)
                    eval_data["val/del_rate"] = overall.get("del_rate", 0.0)
                    eval_data["val/ins_rate"] = overall.get("ins_rate", 0.0)
                    eval_data["left_cer"] = err_raw.get("left_hand_keys", {}).get("CER", 0.0)
                    eval_data["right_cer"] = err_raw.get("right_hand_keys", {}).get("CER", 0.0)
            except Exception:
                pass

        records.append({
            "exp_id": exp_id,
            "seed": seed,
            "folder": exp_folder.name,
            "hw": hw_data,
            "eval": eval_data,
        })
    return records

def format_mean_std(values: list[float], percent: bool = True, ddof: int = 1) -> str:
    if not values:
        return "N/A"
    arr = np.array(values)
    mean = np.mean(arr)
    # If single value, std is 0
    std = np.std(arr, ddof=ddof) if len(arr) > 1 else 0.0
    if percent:
        return f"{mean*100:.2f} ± {std*100:.2f}%"
    return f"{mean:.2f} ± {std:.2f}"

def aggregate_and_print(records: list[dict]):
    if not records:
        print("No completed experiment records found in results/.")
        return

    grouped = defaultdict(list)
    for r in records:
        grouped[r["exp_id"]].append(r)

    print("\n" + "=" * 90)
    print("UCLA ECE C147A / C247A FINAL PROJECT: EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Experiment ID':<22} | {'Seeds':<7} | {'Val CER':<18} | {'Sub%':<10} | {'Del%':<10} | {'Ins%':<10} | {'Train Time':<12}")
    print("-" * 90)

    latex_rows = []

    for exp_id, runs in sorted(grouped.items()):
        seeds = [r["seed"] for r in runs]
        val_cers = [r["eval"].get("val/CER", 0.0) for r in runs if "val/CER" in r["eval"]]
        subs = [r["eval"].get("val/sub_rate", 0.0) for r in runs if "val/sub_rate" in r["eval"]]
        dels = [r["eval"].get("val/del_rate", 0.0) for r in runs if "val/del_rate" in r["eval"]]
        inss = [r["eval"].get("val/ins_rate", 0.0) for r in runs if "val/ins_rate" in r["eval"]]
        times = [r["hw"].get("training_time_minutes", 0.0) for r in runs if "training_time_minutes" in r["hw"]]

        cer_str = format_mean_std(val_cers) if val_cers else "Running/Pending"
        sub_str = format_mean_std(subs) if subs else "--"
        del_str = format_mean_std(dels) if dels else "--"
        ins_str = format_mean_std(inss) if inss else "--"
        time_str = f"{np.mean(times):.1f} min" if times else "--"
        seeds_str = ",".join(str(s) for s in sorted(seeds))

        print(f"{exp_id:<22} | {seeds_str:<7} | {cer_str:<18} | {sub_str:<10} | {del_str:<10} | {ins_str:<10} | {time_str:<12}")

        if val_cers:
            latex_rows.append(
                f"{exp_id.replace('_', ' ')} & {format_mean_std(val_cers)} & {format_mean_std(subs)} & {format_mean_std(dels)} & {format_mean_std(inss)} & {time_str} \\\\"
            )

    print("-" * 90)
    print("\n--- NEURIPS 2024 LATEX TABLE SNIPPET ---")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Neural Decoding Architectures and Channel Ablation Schemes on Subject \\#89335547.}")
    print("\\label{tab:main_results}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("Configuration & Val CER (\\%) & Sub (\\%) & Del (\\%) & Ins (\\%) & Train Time \\\\")
    print("\\midrule")
    for row in latex_rows:
        print(row)
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}\n")

def main():
    results_dir = PROJECT_ROOT / "results"
    records = load_metrics(results_dir)
    aggregate_and_print(records)

if __name__ == "__main__":
    main()
