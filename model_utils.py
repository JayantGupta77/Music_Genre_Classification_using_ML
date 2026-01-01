# model_utils.py
"""
Drop-in replacement model_utils for your Music Genre Classification project.
Option A implementation — matches the extractor & prediction logic used by app.py.

Usage:
 - from model_utils import predict_from_file
 - or run as CLI: python model_utils.py uploads/temp.wav --debug

This file loads these artifacts (from ./artifacts):
 - model.pkl (required)
 - scaler.pkl (required)
 - encoder.pkl (optional)
 - feature_order.json (required)

The predict_from_file function:
 - builds a single-slice feature vector (duration=3s) using librosa
 - aligns fields against feature_order.json (alias/fuzzy matching)
 - scales via scaler.pkl and predicts with model.pkl
 - returns a dict with keys: genre, confidence, coverage, used_user_extractor, note

This file intentionally mirrors the app.py logic so CLI/eval_folder.py and the Flask UI behave identically.
"""

import os
import json
import joblib
import traceback
import importlib
from typing import Optional

import numpy as np
import pandas as pd

# lazy import librosa inside functions to keep import time low for other flows

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
ENC_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
FEATURE_ORDER_PATH = os.path.join(ARTIFACTS_DIR, "feature_order.json")

# cached artifacts
_model = None
_scaler = None
_encoder = None
_feature_order = None


def _load_artifacts():
    global _model, _scaler, _encoder, _feature_order
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")
        _scaler = joblib.load(SCALER_PATH)
    if _encoder is None and os.path.exists(ENC_PATH):
        try:
            _encoder = joblib.load(ENC_PATH)
        except Exception:
            _encoder = None
    if _feature_order is None:
        if not os.path.exists(FEATURE_ORDER_PATH):
            raise FileNotFoundError(f"feature_order.json not found: {FEATURE_ORDER_PATH}")
        with open(FEATURE_ORDER_PATH, "r", encoding="utf-8") as f:
            _feature_order = json.load(f)


# ---------------------------
# Fallback feature extractor (same as app.py)
# ---------------------------

def compute_features_fallback(path: str, n_mfcc: int = 20, duration: int = 3):
    """Robust feature extraction using librosa. Returns a dict of feature -> float."""
    import librosa

    y, sr = librosa.load(path, duration=duration, sr=None)
    feats = {}

    # RMS
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
    except Exception:
        feats["rms_mean"] = 0.0; feats["rms_std"] = 0.0

    # Zero crossing
    try:
        z = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"] = float(np.mean(z))
        feats["zero_crossing_rate_std"] = float(np.std(z))
    except Exception:
        feats["zero_crossing_rate_mean"] = 0.0; feats["zero_crossing_rate_std"] = 0.0

    # Spectral centroid / bandwidth / rolloff
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spectral_centroid_mean"] = float(np.mean(sc))
        feats["spectral_centroid_std"] = float(np.std(sc))
    except Exception:
        feats["spectral_centroid_mean"] = 0.0; feats["spectral_centroid_std"] = 0.0

    try:
        sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["spectral_bandwidth_mean"] = float(np.mean(sbw))
        feats["spectral_bandwidth_std"] = float(np.std(sbw))
    except Exception:
        feats["spectral_bandwidth_mean"] = 0.0; feats["spectral_bandwidth_std"] = 0.0

    try:
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["rolloff_mean"] = float(np.mean(roll))
        feats["rolloff_std"] = float(np.std(roll))
    except Exception:
        feats["rolloff_mean"] = 0.0; feats["rolloff_std"] = 0.0

    # Chroma
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_stft_mean"] = float(np.mean(chroma))
        feats["chroma_stft_std"] = float(np.std(chroma))
    except Exception:
        feats["chroma_stft_mean"] = 0.0; feats["chroma_stft_std"] = 0.0

    # Harmony (tonnetz) and perceptual (spectral contrast)
    try:
        harm = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        feats["harmony_mean"] = float(np.mean(harm))
        feats["harmony_std"] = float(np.std(harm))
    except Exception:
        feats["harmony_mean"] = 0.0; feats["harmony_std"] = 0.0

    try:
        perc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats["perceptr_mean"] = float(np.mean(perc))
        feats["perceptr_std"] = float(np.std(perc))
    except Exception:
        feats["perceptr_mean"] = 0.0; feats["perceptr_std"] = 0.0

    # Tempo
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, tuple, np.ndarray)):
            tempo = float(np.asarray(tempo).ravel()[0])
        feats["tempo"] = float(tempo)
    except Exception:
        feats["tempo"] = 0.0

    # MFCC mean / std
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = np.mean(mfcc.T, axis=0)
        mfcc_stds = np.std(mfcc.T, axis=0)
        for i in range(len(mfcc_means)):
            feats[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
            feats[f"mfcc{i+1}_std"] = float(mfcc_stds[i])
    except Exception:
        for i in range(n_mfcc):
            feats[f"mfcc{i+1}_mean"] = 0.0
            feats[f"mfcc{i+1}_std"] = 0.0

    # keep older naming variants as zeros if missing
    feats.setdefault("harmony_mean", 0.0)
    feats.setdefault("harmony_std", 0.0)
    feats.setdefault("perceptr_mean", feats.get("perceptr_mean", 0.0))
    feats.setdefault("perceptr_std", feats.get("perceptr_std", 0.0))

    return feats


# ---------------------------
# Feature vector builder (aliasing & fuzzy matching)
# ---------------------------
import re

def build_feature_vector_from_dict(feature_order_list, feature_dict):
    """Return np.array vector aligned with feature_order_list. Uses aliasing & fuzzy matching."""
    vec = np.zeros(len(feature_order_list), dtype=float)

    # normalize keys to lowercase and numeric values where possible
    fd = {}
    for k, v in feature_dict.items():
        if v is None:
            continue
        try:
            fd[k.lower()] = float(v)
        except Exception:
            try:
                fd[k.lower()] = float(np.asarray(v).ravel()[0])
            except Exception:
                pass

    # alias map for common naming diffs
    aliases = {
        "chroma_stft_var": "chroma_stft_std",
        "chroma_var": "chroma_stft_std",
        "chroma_mean": "chroma_stft_mean",
        "perceptr_var": "perceptr_std",
        "percept_var": "perceptr_std",
        "percept_mean": "perceptr_mean",
        "harmony_var": "harmony_std",
        # older naming
        "mfcc1_var": "mfcc1_std",
    }
    aliases = {k.lower(): v.lower() for k, v in aliases.items()}

    for i, name in enumerate(feature_order_list):
        n = name.lower()
        # 1) alias
        if n in aliases:
            ak = aliases[n]
            if ak in fd:
                vec[i] = fd[ak]
                continue
        # 2) exact
        if n in fd:
            vec[i] = fd[n]; continue
        # 3) normalized name
        n2 = re.sub(r"[^a-z0-9]+", "_", n)
        if n2 in fd:
            vec[i] = fd[n2]; continue
        # 4) compact
        n3 = n2.replace("_", "")
        if n3 in fd:
            vec[i] = fd[n3]; continue
        # 5) mfcc variants
        m = re.search(r"mfcc[_\- ]*?(\d+).*?(mean|std|var)?", n)
        if m:
            idx = int(m.group(1))
            suffix = (m.group(2) or "mean").lower()
            if suffix == "var":
                suffix = "std"
            key1 = f"mfcc{idx}_{suffix}"
            key2 = f"mfcc{idx}{suffix}"
            if key1 in fd:
                vec[i] = fd[key1]; continue
            if key2 in fd:
                vec[i] = fd[key2]; continue
        # 6) substring fallback
        for k in fd.keys():
            if k in n:
                try:
                    vec[i] = fd[k]
                    break
                except Exception:
                    pass
    return vec


# ---------------------------
# Try to call user extractor from model_utils module (if present by other code)
# ---------------------------

def try_user_extractor(path: str, duration: int = 3):
    """If the repo defines an extractor function in model_utils, try to call it.
    Accepts functions that take (path) or (y, sr).
    Returns dict/Series/DF/array or None.
    """
    try:
        m = importlib.import_module("model_utils")
    except Exception:
        return None

    cand_names = [n for n in dir(m) if "extract" in n.lower()]
    for name in cand_names:
        func = getattr(m, name)
        # avoid recursion: don't call this module's own compute_features_fallback
        if func is compute_features_fallback or func is build_feature_vector_from_dict:
            continue
        try:
            out = func(path)
            return out
        except TypeError:
            # maybe expects (y, sr)
            try:
                import librosa
                y, sr = librosa.load(path, duration=duration, sr=None)
                out = func(y, sr)
                return out
            except Exception:
                continue
        except Exception:
            continue
    return None


# ---------------------------
# High-level prediction function (mirrors app.py)
# ---------------------------

def predict_from_file(file_path: str, debug: bool = False) -> dict:
    """Predict single file and return dict with keys:
    genre (str), numeric (int/str), confidence (float), coverage (float), used_user_extractor (bool), note
    """
    out = {"genre": None, "numeric": None, "confidence": None, "coverage": None, "used_user_extractor": False, "note": None}
    try:
        _load_artifacts()
    except Exception as e:
        out["note"] = f"artifact load error: {e}"
        if debug:
            print("[DEBUG] artifact load error:", e)
        return out

    # prepare temp feature dict
    feature_dict = None
    used_user = False

    # try user extractor first (unless it's this module itself)
    try:
        user_out = try_user_extractor(file_path)
    except Exception as e:
        user_out = None
        if debug:
            print("[DEBUG] user extractor crash:", e)

    if user_out is not None:
        try:
            if isinstance(user_out, pd.Series):
                feature_dict = user_out.to_dict(); used_user = True
            elif isinstance(user_out, pd.DataFrame):
                feature_dict = user_out.iloc[0].to_dict(); used_user = True
            elif isinstance(user_out, dict):
                feature_dict = user_out; used_user = True
            else:
                arr = np.asarray(user_out).ravel()
                if arr.size == len(_feature_order):
                    feature_dict = {k: float(arr[i]) for i, k in enumerate(_feature_order)}
                    used_user = True
        except Exception:
            feature_dict = None
            used_user = False

    if feature_dict is None:
        # fallback extractor
        try:
            feature_dict = compute_features_fallback(file_path, n_mfcc=20, duration=3)
            used_user = False
        except Exception as e:
            out["note"] = f"feature extraction failed: {e}"
            if debug:
                print("[DEBUG] feature extraction failed:", traceback.format_exc())
            return out

    # build vector aligned to feature_order
    vec = build_feature_vector_from_dict(_feature_order, feature_dict)
    nonzero_idx = np.where(np.abs(vec) > 1e-9)[0]
    coverage = float(len(nonzero_idx) / float(len(vec)))

    # scale and predict
    try:
        X_df = pd.DataFrame([vec], columns=_feature_order)
        X_scaled = _scaler.transform(X_df)
    except Exception as e:
        out["note"] = f"scaler transform failed: {e}"
        if debug:
            print("[DEBUG] scaler transform failed:", traceback.format_exc())
        return out

    try:
        pred_num = _model.predict(X_scaled)[0]
    except Exception as e:
        out["note"] = f"model predict failed: {e}"
        if debug:
            print("[DEBUG] model predict failed:", traceback.format_exc())
        return out

    proba = None; top_conf = None
    if hasattr(_model, "predict_proba"):
        try:
            proba = _model.predict_proba(X_scaled)[0]
            try:
                idx = int(pred_num) if isinstance(pred_num, (int, np.integer)) else int(np.argmax(proba))
                top_conf = float(proba[idx]) if idx < len(proba) else float(np.max(proba))
            except Exception:
                top_conf = float(np.max(proba))
        except Exception:
            proba = None

    # decode label via encoder if possible
    label = pred_num
    if _encoder is not None:
        try:
            label = _encoder.inverse_transform([pred_num])[0]
        except Exception:
            try:
                label = _encoder.inverse_transform([str(pred_num)])[0]
            except Exception:
                # fallback: try mapping via encoder.classes_
                try:
                    classes = getattr(_encoder, "classes_", None)
                    if classes is not None and isinstance(pred_num, (int, np.integer)) and 0 <= int(pred_num) < len(classes):
                        label = classes[int(pred_num)]
                    else:
                        label = pred_num
                except Exception:
                    label = pred_num

    out["genre"] = str(label)
    try:
        out["numeric"] = int(pred_num) if isinstance(pred_num, (int, np.integer)) else str(pred_num)
    except Exception:
        out["numeric"] = str(pred_num)

    if top_conf is not None:
        out["confidence"] = float(top_conf)
    out["coverage"] = float(coverage)
    out["used_user_extractor"] = bool(used_user)

    if debug:
        if proba is not None:
            print("[DEBUG] probabilities (slice):", np.round(proba[:min(20, len(proba))], 3).tolist())
        print("Sending prediction:", pred_num, "->", label, "confidence:", out.get("confidence"))
        print(f"[DEBUG] feature_order (first 20): {_feature_order[:20]}")
        print(f"[DEBUG] feature_vector slice (first 20): {np.round(vec[:20],4)}")
        print(f"[DEBUG] nonzero count: {len(nonzero_idx)} / {len(vec)} -> coverage: {coverage:.2f}")

    return out


# ---------------------------
# CLI helper
# ---------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("file", help="audio file to predict")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    res = predict_from_file(args.file, debug=args.debug)
    print(json.dumps(res, indent=2))
