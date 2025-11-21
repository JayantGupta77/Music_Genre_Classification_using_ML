# eval_folder.py
import os, joblib, json
from pathlib import Path
from pprint import pprint

BASE = Path(__file__).parent
ART = BASE / "artifacts"
DATA = BASE / "data" / "genres_original"

# model_utils.predict_from_file exists in your repo (we'll reuse it)
import model_utils

def find_genre_folders(root):
    if not root.exists():
        print("Genres folder not found:", root)
        return []
    return [p for p in root.iterdir() if p.is_dir()]

def pick_one_file(folder):
    # pick a file with .wav (first)
    for ext in (".wav", ".WAV"):
        files = sorted([p for p in folder.glob(f"*{ext}")])
        if files:
            return files[0]
    return None

def main():
    folders = find_genre_folders(DATA)
    if not folders:
        print("No genre subfolders found under", DATA)
        return

    results = []
    summary = {}
    for g in sorted(folders):
        true_genre = g.name
        sample = pick_one_file(g)
        if sample is None:
            print("No sample wav in", g)
            continue
        try:
            pred = model_utils.predict_from_file(str(sample))
        except Exception as e:
            print("Error predicting", sample, e)
            pred = {"genre": None, "confidence": None}

        pred_genre = pred.get("genre")
        conf = pred.get("confidence", None)
        matched = (pred_genre == true_genre)
        results.append((true_genre, str(sample.name), pred_genre, conf, matched))
        summary.setdefault(true_genre, {"total": 0, "correct": 0})
        summary[true_genre]["total"] += 1
        if matched:
            summary[true_genre]["correct"] += 1

    print("\nPer-file predictions (one sample per genre):")
    for t, f, p, c, m in results:
        print(f"- {t:10s} | file: {f:20s} | pred: {str(p):10s} | conf: {str(c):6s} | correct: {m}")

    print("\nSummary by genre:")
    for g, v in sorted(summary.items()):
        total = v["total"]
        corr = v["correct"]
        pct = (corr/total)*100 if total else 0.0
        print(f"- {g:10s} : {corr}/{total} correct  ({pct:.1f}%)")

if __name__ == "__main__":
    main()
