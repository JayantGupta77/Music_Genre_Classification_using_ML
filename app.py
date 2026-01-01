from flask import Flask, request, jsonify, render_template
import joblib
import librosa
import numpy as np
import json
import traceback
import os
import re
import pandas as pd
import importlib
from typing import Optional

app = Flask(__name__)

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.pkl")
SCALER_PATH = os.path.join(ART_DIR, "scaler.pkl")
ENC_PATH = os.path.join(ART_DIR, "encoder.pkl")
FEATURE_ORDER_PATH = os.path.join(ART_DIR, "feature_order.json")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

try:
    encoder = joblib.load(ENC_PATH)
except Exception:
    encoder = None
    print("[WARN] encoder.pkl not found or failed to load. Predictions will be numeric or raw.")

with open(FEATURE_ORDER_PATH, "r") as f:
    feature_order = json.load(f)

try:
    print("[STARTUP] model type:", type(model))
    if hasattr(model, "classes_"):
        print("[STARTUP] model.classes_ (len):", len(model.classes_), "sample:", getattr(model, "classes_")[:10])
except Exception:
    pass

try:
    feat_in = getattr(scaler, "feature_names_in_", None)
    print("[STARTUP] scaler loaded. has feature_names_in_:", feat_in is not None)
    if feat_in is not None:
        try:
            print("[STARTUP] scaler.feature_names_in_ (first 20):", list(feat_in)[:20])
        except Exception:
            pass
except Exception:
    pass

print("[STARTUP] feature_order length:", len(feature_order))
print("[STARTUP] feature_order (first 20):", feature_order[:20])

def try_user_extractor(path: str, duration: int = 3):
    
    try:
        m = importlib.import_module("model_utils")
    except Exception:
        return None

    cand_names = [n for n in dir(m) if "extract" in n.lower()]
    for name in cand_names:
        func = getattr(m, name)
        try:
            out = func(path)
            return out
        except TypeError:
            # maybe expects (y, sr)
            try:
                y, sr = librosa.load(path, duration=duration, sr=None)
                out = func(y, sr)
                return out
            except Exception:
                continue
        except Exception:
            continue
    return None

def compute_features_fallback(path: str, n_mfcc: int = 20, duration: int = 3):
    """
    Robust fallback extractor using librosa. Returns dict feature_name -> float.
    Produces a comprehensive set of features expected by the model.
    """
    y, sr = librosa.load(path, duration=duration, sr=None)
    feats = {}

    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
        feats["rms_std"] = float(np.std(rms))
        feats["rms_var"] = float(np.var(rms))
    except Exception:
        feats["rms_mean"] = 0.0; feats["rms_std"] = 0.0; feats["rms_var"] = 0.0

    try:
        z = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"] = float(np.mean(z))
        feats["zero_crossing_rate_std"] = float(np.std(z))
        feats["zero_crossing_rate_var"] = float(np.var(z))
    except Exception:
        feats["zero_crossing_rate_mean"] = 0.0; feats["zero_crossing_rate_std"] = 0.0; feats["zero_crossing_rate_var"] = 0.0

    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spectral_centroid_mean"] = float(np.mean(sc))
        feats["spectral_centroid_std"] = float(np.std(sc))
        feats["spectral_centroid_var"] = float(np.var(sc))
    except Exception:
        feats["spectral_centroid_mean"] = 0.0; feats["spectral_centroid_std"] = 0.0; feats["spectral_centroid_var"] = 0.0

    try:
        sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["spectral_bandwidth_mean"] = float(np.mean(sbw))
        feats["spectral_bandwidth_std"] = float(np.std(sbw))
        feats["spectral_bandwidth_var"] = float(np.var(sbw))
    except Exception:
        feats["spectral_bandwidth_mean"] = 0.0; feats["spectral_bandwidth_std"] = 0.0; feats["spectral_bandwidth_var"] = 0.0

    try:
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["rolloff_mean"] = float(np.mean(roll))
        feats["rolloff_std"] = float(np.std(roll))
        feats["rolloff_var"] = float(np.var(roll))
    except Exception:
        feats["rolloff_mean"] = 0.0; feats["rolloff_std"] = 0.0; feats["rolloff_var"] = 0.0

    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_stft_mean"] = float(np.mean(chroma))
        feats["chroma_stft_std"] = float(np.std(chroma))
        feats["chroma_stft_var"] = float(np.var(chroma))
    except Exception:
        feats["chroma_stft_mean"] = 0.0; feats["chroma_stft_std"] = 0.0; feats["chroma_stft_var"] = 0.0

    try:
        harm = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        feats["harmony_mean"] = float(np.mean(harm))
        feats["harmony_std"] = float(np.std(harm))
        feats["harmony_var"] = float(np.var(harm))
    except Exception:
        feats["harmony_mean"] = 0.0; feats["harmony_std"] = 0.0; feats["harmony_var"] = 0.0

    try:
        perc = librosa.feature.spectral_contrast(y=y, sr=sr)
        feats["perceptr_mean"] = float(np.mean(perc))
        feats["perceptr_std"] = float(np.std(perc))
        feats["perceptr_var"] = float(np.var(perc))
    except Exception:
        feats["perceptr_mean"] = 0.0; feats["perceptr_std"] = 0.0; feats["perceptr_var"] = 0.0

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, tuple, np.ndarray)):
            tempo = float(np.asarray(tempo).ravel()[0])
        feats["tempo"] = float(tempo)
    except Exception:
        feats["tempo"] = 0.0

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_means = np.mean(mfcc.T, axis=0)
        mfcc_stds = np.std(mfcc.T, axis=0)
        for i in range(len(mfcc_means)):
            feats[f"mfcc{i+1}_mean"] = float(mfcc_means[i])
            feats[f"mfcc{i+1}_std"] = float(mfcc_stds[i])
            feats[f"mfcc{i+1}_var"] = float(mfcc_stds[i])
    except Exception:
        for i in range(n_mfcc):
            feats[f"mfcc{i+1}_mean"] = 0.0
            feats[f"mfcc{i+1}_std"] = 0.0
            feats[f"mfcc{i+1}_var"] = 0.0

    return feats

def build_feature_vector_from_dict(feature_order_list, feature_dict):
    """
    Returns np.array vector (1D) matching feature_order_list.
    Uses alias mapping to maximize coverage.
    """
    vec = np.zeros(len(feature_order_list), dtype=float)

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

    aliases = {
        "chroma_stft_var": "chroma_stft_std",
        "chroma_var": "chroma_stft_std",
        "perceptr_var": "perceptr_std",
        "percept_var": "perceptr_std",
        "percept_mean": "perceptr_mean",
        "perceptr_mean": "perceptr_mean",
        "harmony_var": "harmony_std",
        "rms_var": "rms_std",
    }
    aliases = {k.lower(): v.lower() for k, v in aliases.items()}

    for i, name in enumerate(feature_order_list):
        n = name.lower()
        # alias
        if n in aliases:
            alias_key = aliases[n]
            if alias_key in fd:
                vec[i] = fd[alias_key]
                continue

        if n in fd:
            vec[i] = fd[n]
            continue

        n2 = re.sub(r"[^a-z0-9]+", "_", n)
        if n2 in fd:
            vec[i] = fd[n2]
            continue

        n3 = n2.replace("_", "")
        if n3 in fd:
            vec[i] = fd[n3]
            continue

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

        for k in fd.keys():
            if k in n:
                try:
                    vec[i] = fd[k]
                    break
                except Exception:
                    pass
    return vec

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/debug_features", methods=["POST"])
def debug_features():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", "temp.wav")
        file.save(temp_path)

        user_out = None
        try:
            user_out = try_user_extractor(temp_path, duration=3)
        except Exception as e:
            user_out = None
            print("[WARN] user extractor error in debug_features:", e)

        feature_dict = None
        used_user = False
        if user_out is not None:
            if isinstance(user_out, pd.Series):
                feature_dict = user_out.to_dict(); used_user = True
            elif isinstance(user_out, pd.DataFrame):
                feature_dict = user_out.iloc[0].to_dict(); used_user = True
            elif isinstance(user_out, dict):
                feature_dict = user_out; used_user = True
            else:
                arr = np.asarray(user_out).ravel()
                if arr.size == len(feature_order):
                    feature_dict = {k: float(arr[i]) for i, k in enumerate(feature_order)}
                    used_user = True

        if feature_dict is None:
            feature_dict = compute_features_fallback(temp_path, n_mfcc=20, duration=3)
            used_user = False

        vec = build_feature_vector_from_dict(feature_order, feature_dict)
        nonzero_idx = np.where(np.abs(vec) > 1e-9)[0]
        coverage = float(len(nonzero_idx) / len(vec))

        map_report = []
        for i in nonzero_idx[:60]:
            map_report.append({"idx": int(i), "feature_order": feature_order[i], "value": float(vec[i])})

        resp = {
            "used_user_extractor": bool(used_user),
            "feature_count_expected": len(feature_order),
            "nonzero_count": int(len(nonzero_idx)),
            "coverage": coverage,
            "mapping_sample": map_report,
            "feature_dict_sample": {k: feature_dict.get(k, None) for k in list(feature_dict.keys())[:80]}
        }
        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty file"}), 400

        os.makedirs("uploads", exist_ok=True)
        temp_path = os.path.join("uploads", "temp.wav")
        file.save(temp_path)
        print("[INFO] saved upload ->", temp_path)

        used_user = False
        user_out = None
        if request.args.get("force_fallback", "0").lower() not in ("1", "true", "yes"):
            try:
                user_out = try_user_extractor(temp_path, duration=3)
            except Exception as e:
                print("[WARN] user extractor crashed:", e)
                user_out = None

        feature_dict = None
        vec = None

        if user_out is not None:
            try:
                if isinstance(user_out, pd.Series):
                    feature_dict = user_out.to_dict()
                    vec = build_feature_vector_from_dict(feature_order, feature_dict)
                    used_user = True
                elif isinstance(user_out, pd.DataFrame):
                    feature_dict = user_out.iloc[0].to_dict()
                    vec = build_feature_vector_from_dict(feature_order, feature_dict)
                    used_user = True
                elif isinstance(user_out, dict):
                    feature_dict = user_out
                    vec = build_feature_vector_from_dict(feature_order, feature_dict)
                    used_user = True
                else:
                    arr = np.asarray(user_out).ravel()
                    if arr.size == len(feature_order):
                        vec = arr.astype(float)
                        used_user = True
            except Exception as e:
                print("[WARN] interpreting user extractor output failed:", e)
                used_user = False

        if vec is None:
            print("[INFO] Using fallback extractor")
            feature_dict = compute_features_fallback(temp_path, n_mfcc=20, duration=3)
            vec = build_feature_vector_from_dict(feature_order, feature_dict)
            used_user = False

        nonzero_idx = np.where(np.abs(vec) > 1e-9)[0]
        coverage = len(nonzero_idx) / float(len(vec))
        print(f"[DEBUG] feature_order (first 20): {feature_order[:20]}")
        print(f"[DEBUG] feature_vector slice (first 20): {np.round(vec[:20], 4)}")
        print(f"[DEBUG] nonzero count: {len(nonzero_idx)} / {len(vec)} -> coverage: {coverage:.2f}")

        X_df = pd.DataFrame([vec], columns=feature_order)
        X_scaled = scaler.transform(X_df)
        pred_num = model.predict(X_scaled)[0]

        proba = None; top_conf = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0]
            try:
                idx = int(pred_num) if isinstance(pred_num, (int, np.integer)) else int(np.argmax(proba))
                top_conf = float(proba[idx]) if idx < len(proba) else float(np.max(proba))
            except Exception:
                top_conf = float(np.max(proba))
            print("[DEBUG] probabilities (slice):", np.round(proba[:8], 3).tolist())

        label = pred_num
        if encoder is not None:
            try:
                label = encoder.inverse_transform([pred_num])[0]
            except Exception:
                try:
                    label = encoder.inverse_transform([str(pred_num)])[0]
                except Exception:
                    try:
                        classes = getattr(encoder, "classes_", None)
                        if classes is not None and isinstance(pred_num, (int, np.integer)) and 0 <= int(pred_num) < len(classes):
                            label = classes[int(pred_num)]
                        else:
                            label = pred_num
                    except Exception:
                        label = pred_num

        print("Sending prediction:", pred_num, "->", label, " confidence:", top_conf)
        if proba is not None:
            print("[DEBUG] probabilities (slice):", np.round(proba[:8], 3).tolist())

        resp = {"genre": str(label), "numeric": int(pred_num) if isinstance(pred_num, (int, np.integer)) else str(pred_num)}
        if top_conf is not None:
            resp["confidence"] = float(top_conf)
        resp["coverage"] = float(coverage)
        resp["used_user_extractor"] = bool(used_user)

        return jsonify(resp)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
