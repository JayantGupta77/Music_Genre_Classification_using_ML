# eval_folder_updated.py
"""
Evaluate predictions on a folder of .wav files.
Each file name must follow the format: <genre>.<anything>.wav
Example: blues.00000.wav

This script calls model_utils.predict_from_file() correctly
(only supports the arguments: file_path, debug=False)

Output:
 - per-file predictions
 - summary accuracy by genre
 - overall summary
"""

import os
import argparse
from model_utils import predict_from_file


def extract_genre_from_filename(filename):
    """
    Extracts the genre from a filename of format:
    blues.00003.wav → "blues"
    hiphop.xyz.wav → "hiphop"
    """
    parts = filename.split(".")
    if len(parts) >= 2:
        return parts[0].lower().strip()
    return None


def evaluate_folder(folder, debug=False):
    if not os.path.isdir(folder):
        print(f"[ERROR] Folder not found: {folder}")
        return

    print(f"\n=== Evaluating folder: {folder} ===\n")

    files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(".wav")]

    if not files:
        print("[WARN] No .wav files found in this folder.")
        return

    results = {}  # genre → list of booleans

    for f in files:
        true_genre = extract_genre_from_filename(f)
        full_path = os.path.join(folder, f)

        if true_genre not in results:
            results[true_genre] = []

        print(f"- {true_genre:<10} | file: {f}")

        try:
            pred = predict_from_file(full_path, debug=debug)

            predicted_genre = pred.get("genre")
            conf = pred.get("confidence")
            cov = pred.get("coverage")

            is_correct = (predicted_genre == true_genre)

            print(f"    → predicted: {predicted_genre} | "
                  f"conf: {conf} | cov: {cov} | correct: {is_correct}")

            results[true_genre].append(is_correct)

        except Exception as e:
            print(f"[ERROR] Failed on file {f}: {e}")
            results[true_genre].append(False)

    print("\n=== SUMMARY BY GENRE ===")
    total_correct = 0
    total_files = 0

    for g, lst in results.items():
        correct = sum(lst)
        count = len(lst)
        total_correct += correct
        total_files += count
        acc = (correct / count * 100) if count > 0 else 0
        print(f"- {g:<10}: {correct}/{count} correct ({acc:.1f}%)")

    overall = (total_correct / total_files * 100) if total_files > 0 else 0
    print("\n=== OVERALL SUMMARY ===")
    print(f"Total correct: {total_correct}/{total_files}  ({overall:.1f}%)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data/test_one_each",
                        help="Folder containing one test file per genre.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output from predict_from_file.")
    args = parser.parse_args()

    evaluate_folder(args.folder, debug=args.debug)
