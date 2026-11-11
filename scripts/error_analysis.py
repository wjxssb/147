#!/usr/bin/env python3
"""
scripts/error_analysis.py
==========================
Fine-grained Character Error Rate (CER) and Biomechanical Key Breakdown.

Key features:
1. Levenshtein Dynamic Programming Backtracking:
   Extracts exact edit operations (Match, Substitution, Deletion, Insertion).
2. QWERTY Biomechanical Spatial Breakdown:
   - Left-Hand Keys: Q, W, E, R, T, A, S, D, F, G, Z, X, C, V (and uppercase)
   - Right-Hand Keys: Y, U, I, O, P, H, J, K, L, N, M (and uppercase)
   - Neutral / Special Keys: Space, Numbers, Punctuation
3. Attribution Protocol:
   - Substitutions & Deletions: Assigned according to ground-truth reference character's key zone.
   - Insertions: Categorized based on the predicted character's key zone (or reported separately),
     preserving strict statistical honesty without arbitrary hand assignment.
4. Error Breakdown:
   Computes individual Substitution Rate, Deletion Rate, and Insertion Rate.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

LEFT_HAND_KEYS = set("qwertasdfgzxcvQWERTASDFGZXCV")
RIGHT_HAND_KEYS = set("yuiophjklbnmYUIOPHJKLBNM")

def levenshtein_alignment(ref: str, hyp: str):
    """
    Computes Levenshtein matrix and backtracks to find the optimal alignment operations.
    Returns: list of tuples (op, ref_char, hyp_char)
      op in {'M' (Match), 'S' (Sub), 'D' (Del), 'I' (Ins)}
    """
    N = len(ref)
    M = len(hyp)
    dp = [[0] * (M + 1) for _ in range(N + 1)]

    for i in range(N + 1):
        dp[i][0] = i
    for j in range(M + 1):
        dp[0][j] = j

    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if ref[i - 1] == hyp[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # Deletion
                dp[i][j - 1] + 1,       # Insertion
                dp[i - 1][j - 1] + cost # Match or Substitution
            )

    # Backtrack
    i, j = N, M
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == hyp[j - 1] else 1):
            op = "M" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append((op, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], None))
            i -= 1
        else:
            ops.append(("I", None, hyp[j - 1]))
            j -= 1

    ops.reverse()
    return dp[N][M], ops

def categorize_key(char: str | None) -> str:
    if char is None:
        return "none"
    if char in LEFT_HAND_KEYS:
        return "left"
    elif char in RIGHT_HAND_KEYS:
        return "right"
    else:
        return "neutral"

def analyze_predictions(predictions: list[dict]):
    """
    Analyzes a list of sample dicts: [{'ref': str, 'hyp': str}, ...]
    Computes global metrics and biomechanical left/right key breakdown.
    """
    total_ref_chars = 0
    total_sub = 0
    total_del = 0
    total_ins = 0

    hand_stats = {
        "left": {"ref_count": 0, "sub": 0, "del": 0, "ins": 0},
        "right": {"ref_count": 0, "sub": 0, "del": 0, "ins": 0},
        "neutral": {"ref_count": 0, "sub": 0, "del": 0, "ins": 0},
    }

    confusion_matrix = defaultdict(lambda: defaultdict(int))

    for item in predictions:
        ref = item["ref"]
        hyp = item["hyp"]
        edit_dist, ops = levenshtein_alignment(ref, hyp)

        for op, r_char, h_char in ops:
            if op == "M":
                total_ref_chars += 1
                hand = categorize_key(r_char)
                hand_stats[hand]["ref_count"] += 1
            elif op == "S":
                total_ref_chars += 1
                total_sub += 1
                hand = categorize_key(r_char)
                hand_stats[hand]["ref_count"] += 1
                hand_stats[hand]["sub"] += 1
                confusion_matrix[r_char][h_char] += 1
            elif op == "D":
                total_ref_chars += 1
                total_del += 1
                hand = categorize_key(r_char)
                hand_stats[hand]["ref_count"] += 1
                hand_stats[hand]["del"] += 1
                confusion_matrix[r_char]["<DEL>"] += 1
            elif op == "I":
                total_ins += 1
                h_hand = categorize_key(h_char)
                hand_stats[h_hand]["ins"] += 1
                confusion_matrix["<INS>"][h_char] += 1

    overall_cer = (total_sub + total_del + total_ins) / max(total_ref_chars, 1)
    sub_rate = total_sub / max(total_ref_chars, 1)
    del_rate = total_del / max(total_ref_chars, 1)
    ins_rate = total_ins / max(total_ref_chars, 1)

    # Hand-specific error rates
    left_ref = max(hand_stats["left"]["ref_count"], 1)
    right_ref = max(hand_stats["right"]["ref_count"], 1)

    left_cer = (hand_stats["left"]["sub"] + hand_stats["left"]["del"] + hand_stats["left"]["ins"]) / left_ref
    right_cer = (hand_stats["right"]["sub"] + hand_stats["right"]["del"] + hand_stats["right"]["ins"]) / right_ref

    results = {
        "overall": {
            "total_ref_chars": total_ref_chars,
            "CER": overall_cer,
            "sub_rate": sub_rate,
            "del_rate": del_rate,
            "ins_rate": ins_rate,
            "substitutions": total_sub,
            "deletions": total_del,
            "insertions": total_ins,
        },
        "left_hand_keys": {
            "ref_chars": hand_stats["left"]["ref_count"],
            "CER": left_cer,
            "sub": hand_stats["left"]["sub"],
            "del": hand_stats["left"]["del"],
            "ins": hand_stats["left"]["ins"],
        },
        "right_hand_keys": {
            "ref_chars": hand_stats["right"]["ref_count"],
            "CER": right_cer,
            "sub": hand_stats["right"]["sub"],
            "del": hand_stats["right"]["del"],
            "ins": hand_stats["right"]["ins"],
        },
        "neutral_keys": {
            "ref_chars": hand_stats["neutral"]["ref_count"],
            "sub": hand_stats["neutral"]["sub"],
            "del": hand_stats["neutral"]["del"],
            "ins": hand_stats["neutral"]["ins"],
        },
    }
    return results, confusion_matrix

def main():
    parser = argparse.ArgumentParser(description="Detailed CER & Levenshtein Error Analysis")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to predictions JSON/JSONL")
    parser.add_argument("--out_file", type=str, default="error_analysis.json", help="Path to output JSON")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    if not pred_path.exists():
        print(f"Error: {pred_path} does not exist")
        sys.exit(1)

    with open(pred_path) as f:
        data = json.load(f)

    metrics, confusion = analyze_predictions(data)
    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"Total Ref Chars: {metrics['overall']['total_ref_chars']}")
    print(f"Overall CER:     {metrics['overall']['CER']*100:.2f}% (Sub: {metrics['overall']['sub_rate']*100:.2f}%, Del: {metrics['overall']['del_rate']*100:.2f}%, Ins: {metrics['overall']['ins_rate']*100:.2f}%)")
    print(f"Left-Hand CER:   {metrics['left_hand_keys']['CER']*100:.2f}% (Ref Chars: {metrics['left_hand_keys']['ref_chars']})")
    print(f"Right-Hand CER:  {metrics['right_hand_keys']['CER']*100:.2f}% (Ref Chars: {metrics['right_hand_keys']['ref_chars']})")

    with open(args.out_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Detailed metrics saved to {args.out_file}")

if __name__ == "__main__":
    main()
