#!/usr/bin/env python3
"""
Evaluate the model on a folder of genre subfolders (e.g. GTZAN layout).
Saves per-file predictions to CSV and prints a confusion matrix + classification report.

Usage:
  python eval_full_dataset.py --folder data/genres_original --out results/predictions.csv --max-per-genre 100 --debug
"""
import os
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Import your project extractor/predictor
import model_utils

# sklearn for final metrics
from sklearn.metrics import confusion_matrix, classification_report

def iter_genre_files(root_folder: str):
    root = Path(root_folder)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root_folder}")
    genres = [p for p in sorted(root.iterdir()) if p.is_dir()]
    for g in genres:
        for wav in sorted(g.glob("*.wav")):
            yield g.name, str(wav)

def run_eval(folder, out_csv, max_per_genre=None, debug=False):
    rows = []
    counts = defaultdict(int)
    # create output folder
    out_dir = Path(out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for genre, path in iter_genre_files(folder):
        if max_per_genre and counts[genre] >= int(max_per_genre):
            continue
        counts[genre] += 1

        # Call your model util API
        try:
            res = model_utils.predict_from_file(path, debug=debug)
        except Exception as e:
            if debug:
                print("[WARN] predict_from_file failed for", path, "->", e)
            res = {"genre": None, "confidence": None, "coverage": None, "numeric": None, "note": f"exception:{e}"}

        pred = res.get("genre")
        conf = res.get("confidence")
        cov = res.get("coverage")
        numeric = res.get("numeric", None)
        note = res.get("note", None)

        rows.append({
            "true_genre": genre,
            "file": path,
            "pred_genre": pred,
            "pred_numeric": numeric,
            "confidence": conf,
            "coverage": cov,
            "note": note
        })

        if debug:
            print(f"[DBG] true={genre} pred={pred} conf={conf} cov={cov} file={Path(path).name}")

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} rows to {out_csv}")

    # Prepare labels for metrics: drop rows without a predicted label
    df_valid = df[df["pred_genre"].notna()].copy()
    if df_valid.empty:
        print("No valid predictions to score.")
        return df

    y_true = df_valid["true_genre"].astype(str).tolist()
    y_pred = df_valid["pred_genre"].astype(str).tolist()

    labels = sorted(list(set(y_true) | set(y_pred)))
    print("\n=== Classification report ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    print("\n=== Confusion matrix ===")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    # show per-genre accuracy summary
    summary = {}
    for lbl in labels:
        total = sum(1 for t in y_true if t == lbl)
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == lbl and p == lbl)
        summary[lbl] = {"total": total, "correct": correct, "acc": (correct / total) if total else 0.0}

    print("\n=== Per-genre summary ===")
    for lbl in labels:
        s = summary[lbl]
        print(f"- {lbl:10s} : {s['correct']}/{s['total']} correct  ({s['acc']*100:.1f}%)")

    overall_acc = sum([v["correct"] for v in summary.values()]) / max(1, sum([v["total"] for v in summary.values()]))
    print(f"\nOverall accuracy: {overall_acc*100:.1f}% ({sum([v['correct'] for v in summary.values()])}/{sum([v['total'] for v in summary.values()])})")

    return df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folder", type=str, default="data/genres_original", help="Root folder with genre subfolders")
    p.add_argument("--out", type=str, default="predictions.csv", help="CSV output path")
    p.add_argument("--max-per-genre", type=int, default=None, help="Limit files per genre (for fast tests)")
    p.add_argument("--debug", action="store_true", help="Enable debug prints")
    args = p.parse_args()

    run_eval(args.folder, args.out, max_per_genre=args.max_per_genre, debug=args.debug)
