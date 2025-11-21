# model_utils.py
import os
import json
import joblib
import numpy as np
import librosa
import warnings

warnings.filterwarnings("ignore")

# paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
ENC_PATH = os.path.join(ARTIFACTS_DIR, "encoder.pkl")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
COLS_PATH = os.path.join(ARTIFACTS_DIR, "columns.pkl")         # optional
FEATURE_ORDER_JSON = os.path.join(ARTIFACTS_DIR, "feature_order.json")  # used in your repo

# cached objects
_model = None
_encoder = None
_scaler = None
_columns = None   # list of feature names expected by the model


def _load_artifacts():
    """Load model, encoder, scaler, and column list into module globals."""
    global _model, _encoder, _scaler, _columns

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)

    if _encoder is None and os.path.exists(ENC_PATH):
        _encoder = joblib.load(ENC_PATH)

    if _scaler is None and os.path.exists(SCALER_PATH):
        _scaler = joblib.load(SCALER_PATH)

    # columns could be stored as columns.pkl or feature_order.json
    if _columns is None:
        if os.path.exists(COLS_PATH):
            _columns = joblib.load(COLS_PATH)
        elif os.path.exists(FEATURE_ORDER_JSON):
            try:
                with open(FEATURE_ORDER_JSON, "r", encoding="utf-8") as fh:
                    # file may contain an array of names or a mapping; expect list
                    cols = json.load(fh)
                    if isinstance(cols, dict) and "feature_order" in cols:
                        cols = cols["feature_order"]
                    if not isinstance(cols, list):
                        raise ValueError("feature_order.json did not contain a list")
                    _columns = cols
            except Exception as e:
                raise RuntimeError(f"Unable to read feature order from {FEATURE_ORDER_JSON}: {e}")
        else:
            raise FileNotFoundError("No columns.pkl or feature_order.json found in artifacts.")


# -------------------------
# Feature extraction utils
# -------------------------
def _safe_mean_std(arr):
    """Return mean and std floats for array-like (or (0.0, 0.0) if empty)."""
    if arr is None or len(arr) == 0:
        return 0.0, 0.0
    a = np.asarray(arr, dtype=float)
    return float(np.mean(a)), float(np.std(a))


def extract_features_from_file(path, duration=3, sr=None, n_mfcc=13):
    """
    Extract a set of features from an audio file.
    Returns a dict mapping feature_name -> scalar value.
    Designed to produce keys similar to your training extractor (mfcc1_mean, mfcc1_std, rms_mean, etc).
    """
    feats = {}
    try:
        # load short clip (duration seconds) to match training extractors
        y, fs = librosa.load(path, sr=sr, mono=True, duration=duration)
        if y is None or y.size == 0:
            raise ValueError("Empty audio signal")
    except Exception as e:
        # bubble up a descriptive error
        raise RuntimeError(f"librosa.load failed for {path}: {e}")

    # RMS
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"], feats["rms_std"] = _safe_mean_std(rms)
    except Exception:
        feats["rms_mean"], feats["rms_std"] = 0.0, 0.0

    # Zero crossing rate
    try:
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"], feats["zero_crossing_rate_std"] = _safe_mean_std(zcr)
    except Exception:
        feats["zero_crossing_rate_mean"], feats["zero_crossing_rate_std"] = 0.0, 0.0

    # Spectral centroid
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=fs)[0]
        feats["spectral_centroid_mean"], feats["spectral_centroid_var"] = _safe_mean_std(sc)
    except Exception:
        feats["spectral_centroid_mean"], feats["spectral_centroid_var"] = 0.0, 0.0

    # Spectral bandwidth
    try:
        sb = librosa.feature.spectral_bandwidth(y=y, sr=fs)[0]
        feats["spectral_bandwidth_mean"], feats["spectral_bandwidth_var"] = _safe_mean_std(sb)
    except Exception:
        feats["spectral_bandwidth_mean"], feats["spectral_bandwidth_var"] = 0.0, 0.0

    # Spectral rolloff
    try:
        roll = librosa.feature.spectral_rolloff(y=y, sr=fs)[0]
        feats["rolloff_mean"], feats["rolloff_var"] = _safe_mean_std(roll)
    except Exception:
        feats["rolloff_mean"], feats["rolloff_var"] = 0.0, 0.0

    # Chroma (stft)
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=fs)
        chroma_means = chroma.mean(axis=1)
        chroma_vars = chroma.std(axis=1)
        feats["chroma_stft_mean"] = float(np.mean(chroma_means))
        feats["chroma_stft_var"] = float(np.mean(chroma_vars))
    except Exception:
        feats["chroma_stft_mean"], feats["chroma_stft_var"] = 0.0, 0.0

    # Harmonic (approx)
    try:
        harm = librosa.effects.harmonic(y)
        h_mean, h_std = _safe_mean_std(harm)
        feats["harmony_mean"], feats["harmony_var"] = h_mean, h_std
    except Exception:
        feats["harmony_mean"], feats["harmony_var"] = 0.0, 0.0

    # Perceptual features - approximated by spectral_contrast or poly features; keep placeholder if not available
    try:
        perceptr = librosa.feature.spectral_contrast(y=y, sr=fs)
        feats["perceptr_mean"], feats["perceptr_var"] = _safe_mean_std(perceptr.mean(axis=1))
    except Exception:
        feats["perceptr_mean"], feats["perceptr_var"] = 0.0, 0.0

    # Tempo (beat tracker)
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=fs)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=fs)
        feats["tempo"] = float(tempo[0]) if len(tempo) > 0 else 0.0
    except Exception:
        feats["tempo"] = 0.0

    # MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=n_mfcc)
        for i in range(mfcc.shape[0]):
            idx = i + 1
            feats[f"mfcc{idx}_mean"] = float(np.mean(mfcc[i]))
            feats[f"mfcc{idx}_std"] = float(np.std(mfcc[i]))
    except Exception:
        # create zeros for 13 mfccs if absent
        for i in range(1, n_mfcc + 1):
            feats[f"mfcc{i}_mean"], feats[f"mfcc{i}_std"] = 0.0, 0.0

    return feats


# -------------------------
# feature vector builder
# -------------------------
def build_feature_vector_from_dict(feature_order, feature_dict):
    """
    Given feature_order (list) and feature_dict (map name->value),
    return np.array shape (1, n_features), plus coverage info and missing list.
    """
    vec = []
    missing = []
    nonzero = 0
    total = len(feature_order)
    for k in feature_order:
        if k in feature_dict:
            v = feature_dict[k]
            try:
                val = float(v)
            except Exception:
                val = 0.0
            vec.append(val)
            if val != 0.0:
                nonzero += 1
        else:
            vec.append(0.0)
            missing.append(k)
    coverage = float(nonzero) / float(total) if total > 0 else 0.0
    X = np.array(vec, dtype=float).reshape(1, -1)
    return X, missing, coverage


# -------------------------
# prediction API
# -------------------------
def predict_from_file(file_path, debug=True):
    """
    High-level function used by app.py:
    - loads artifacts
    - extracts features
    - builds feature vector aligned to expected feature list
    - scales (if scaler present)
    - predicts, returns dict with:
        {
           "genre": label,
           "confidence": 0.32,
           "probabilities": {"class_name": prob, ...},
           "coverage": 0.52,
           "missing": [ ... ]
        }
    """
    global _columns
    _load_artifacts()

    # extract features
    feat_dict = extract_features_from_file(file_path)

    # if model expects columns in _columns variable (loaded from file)
    feature_order = _columns
    if feature_order is None:
        raise RuntimeError("Feature order (columns) not loaded; cannot build vector.")

    # build vector (aligned)
    X_raw, missing, coverage = build_feature_vector_from_dict(feature_order, feat_dict)

    if debug:
        print("[DEBUG] feature_order (first 20):", feature_order[:20])
        print("[DEBUG] feature_vector slice (first 20):", np.round(X_raw[0, :min(20, X_raw.shape[1])], 6))
        print(f"[DEBUG] nonzero count: {np.count_nonzero(X_raw)} / {X_raw.size} -> coverage: {coverage:.2f}")

    # optionally scale (scaler from artifacts)
    X = X_raw
    if _scaler is not None:
        try:
            X = _scaler.transform(X_raw)
            if debug:
                print("[DEBUG] applied scaler")
        except Exception as e:
            # don't fail prediction entirely if scaler transform fails; issue debug note
            if debug:
                print("[DEBUG] scaler.transform failed:", e)
            X = X_raw

    # predict
    model = _model
    if model is None:
        raise RuntimeError("Model not loaded")

    # predict label numeric or encoded
    try:
        pred_num = model.predict(X)
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

    # prepare probability vector if available
    probs = None
    try:
        if hasattr(model, "predict_proba"):
            probs_arr = model.predict_proba(X)[0]  # (n_classes,)
            probs = {str(cls): float(p) for cls, p in zip(getattr(model, "classes_", range(len(probs_arr))), probs_arr)}
            if debug:
                print("[DEBUG] probabilities (slice):", np.round(probs_arr[:10], 3).tolist())
    except Exception as e:
        if debug:
            print("[DEBUG] predict_proba failed:", e)
        probs = None

    # decide label string
    label = pred_num[0] if isinstance(pred_num, (list, np.ndarray)) else pred_num
    # if an encoder is present, attempt to inverse transform
    final_label = None
    if _encoder is not None:
        try:
            # encoder may expect shape (n_samples,)
            inv = _encoder.inverse_transform(np.atleast_1d(label))
            final_label = inv[0]
        except Exception:
            # fallback: if model.classes_ exists, map numeric to model.classes_
            try:
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    final_label = classes[int(label)]
                else:
                    final_label = str(label)
            except Exception:
                final_label = str(label)
    else:
        # no encoder: try model.classes_ mapping
        classes = getattr(model, "classes_", None)
        if classes is not None:
            try:
                final_label = classes[int(label)]
            except Exception:
                final_label = str(label)
        else:
            final_label = str(label)

    # confidence: if probs available, use max; else None
    confidence = None
    if probs is not None:
        try:
            confidence = float(max(probs.values()))
        except Exception:
            confidence = None

    # build probabilities as human-friendly mapping of encoder classes if possible
    prob_map = None
    if probs is not None:
        # build mapping from readable class names (if encoder exists)
        prob_map = {}
        # attempt to map model.classes_ to readable labels
        model_classes = getattr(model, "classes_", None)
        if model_classes is not None and _encoder is not None:
            # model_classes are encoded ints -> need inverse_transform
            try:
                readable = list(_encoder.inverse_transform(model_classes))
                for cls_val, rd in zip(model_classes, readable):
                    prob_map[str(rd)] = float(probs.get(str(cls_val), 0.0))
            except Exception:
                # fallback: write probs keyed by model.classes_
                for cls_val in model_classes:
                    prob_map[str(cls_val)] = float(probs.get(str(cls_val), 0.0))
        else:
            for k, v in probs.items():
                prob_map[str(k)] = float(v)

    result = {
        "genre": str(final_label),
        "confidence": round(confidence, 4) if confidence is not None else None,
        "probabilities": prob_map,
        "coverage": round(float(coverage), 4),
        "missing": missing,
        "raw_features": None  # optional: can include compact raw features if needed
    }

    return result


# convenience alias for older code
def predict(file_path, debug=True):
    return predict_from_file(file_path, debug=debug)


# for quick local debug run
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("file", help="audio file to predict")
    args = p.parse_args()
    out = predict_from_file(args.file, debug=True)
    print("PRED:", out)
