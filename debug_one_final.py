# debug_one_final.py
import joblib
import json
import numpy as np
import os
import importlib
import traceback

# try librosa only when needed
try:
    import librosa
except Exception:
    librosa = None

# ---------- helpers (mirrors app logic) ----------
def try_user_extractor(path):
    try:
        m = importlib.import_module("model_utils")
        for fn in ("extract_features","make_features","get_features","compute_features","extract","extract_features_from_file"):
            if hasattr(m, fn):
                func = getattr(m, fn)
                try:
                    out = func(path)
                    return out, f"called:{fn}"
                except TypeError:
                    # maybe expects y,sr
                    try:
                        if librosa is None:
                            return None, "librosa_missing_for_audio_load"
                        y, sr = librosa.load(path, duration=3, sr=None)
                        out = func(y, sr)
                        return out, f"called:{fn}(y,sr)"
                    except Exception as e:
                        continue
        return None, "no_fn_found"
    except Exception as e:
        return None, f"import_failed:{e}"

def compute_features_fallback(path, n_mfcc=20, duration=3):
    if librosa is None:
        raise RuntimeError("librosa not installed in this environment.")
    y, sr = librosa.load(path, duration=duration, sr=None)
    feats = {}
    # rms
    try:
        f = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(f)); feats["rms_std"] = float(np.std(f))
    except Exception:
        feats["rms_mean"]=0.0; feats["rms_std"]=0.0
    # zcr
    try:
        f = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"]=float(np.mean(f)); feats["zero_crossing_rate_std"]=float(np.std(f))
    except Exception:
        feats["zero_crossing_rate_mean"]=0.0; feats["zero_crossing_rate_std"]=0.0
    # spectral centroid/bandwidth/rolloff
    try:
        f = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spectral_centroid_mean"]=float(np.mean(f)); feats["spectral_centroid_std"]=float(np.std(f))
    except Exception:
        feats["spectral_centroid_mean"]=0.0; feats["spectral_centroid_std"]=0.0
    try:
        f = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["spectral_bandwidth_mean"]=float(np.mean(f)); feats["spectral_bandwidth_std"]=float(np.std(f))
    except Exception:
        feats["spectral_bandwidth_mean"]=0.0; feats["spectral_bandwidth_std"]=0.0
    try:
        f = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["rolloff_mean"]=float(np.mean(f)); feats["rolloff_std"]=float(np.std(f))
    except Exception:
        feats["rolloff_mean"]=0.0; feats["rolloff_std"]=0.0
    # chroma
    try:
        f = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_stft_mean"]=float(np.mean(f)); feats["chroma_stft_var"]=float(np.var(f))
    except Exception:
        feats["chroma_stft_mean"]=0.0; feats["chroma_stft_var"]=0.0
    # tempo
    try:
        t, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(t)
    except Exception:
        feats["tempo"] = 0.0
    # mfccs
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        means = np.mean(mfcc.T, axis=0); vars_ = np.var(mfcc.T, axis=0)
        for i in range(len(means)):
            feats[f"mfcc{i+1}_mean"] = float(means[i])
            feats[f"mfcc{i+1}_var"]  = float(vars_[i])
    except Exception:
        for i in range(n_mfcc):
            feats[f"mfcc{i+1}_mean"]=0.0; feats[f"mfcc{i+1}_var"]=0.0

    # small safe placeholders
    feats.setdefault("harmony_mean", 0.0)
    feats.setdefault("harmony_var", 0.0)
    feats.setdefault("perceptr_mean", 0.0)
    feats.setdefault("perceptr_var", 0.0)

    return feats

def build_vector_from_dict(feature_order, feature_dict):
    fd = {k.lower(): float(v) for k,v in feature_dict.items()}
    vec = np.zeros(len(feature_order), dtype=float)
    for i,name in enumerate(feature_order):
        n = name.lower()
        if n in fd:
            vec[i] = fd[n]; continue
        n2 = n.replace("-","_").replace(" ","_")
        if n2 in fd:
            vec[i] = fd[n2]; continue
        n2a = n2.replace("_","")
        if n2a in fd:
            vec[i] = fd[n2a]; continue
        # try mfcc pattern
        import re
        m = re.search(r"(mfcc)[^\d]*(\d+)[^\w]*(mean|std|var)?", n)
        if m:
            base = f"mfcc{int(m.group(2))}_{'mean' if (m.group(3) is None or 'mean' in m.group(3)) else 'var'}"
            if base in fd:
                vec[i] = fd[base]; continue
        # substring fallback
        for k in fd:
            if k in n:
                vec[i] = fd[k]; break
    return vec

# ---------- main debug ----------
def main():
    print("Running debug_one_final.py\n")
    # load artifacts
    art_dir = "artifacts"
    try:
        model = joblib.load(os.path.join(art_dir,"model.pkl"))
        scaler = joblib.load(os.path.join(art_dir,"scaler.pkl"))
    except Exception as e:
        print("Failed to load model/scaler:", e); traceback.print_exc(); return
    try:
        encoder = joblib.load(os.path.join(art_dir,"encoder.pkl"))
    except Exception:
        encoder = None
    with open(os.path.join(art_dir,"feature_order.json"),"r") as f:
        feature_order = json.load(f)
    print("Loaded model, scaler, encoder present?", encoder is not None)
    print("feature_order length:", len(feature_order))
    # show top importances if available
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        print("\nTop 20 feature importances (index -> name -> importance):")
        for i in order[:20]:
            print(f" idx {i:2d} -> {feature_order[i]:20s} -> {importances[i]:0.4f}")
    else:
        print("Model has no feature_importances_. Skipping importance list.")

    # target audio file
    temp = os.path.join("uploads","temp.wav")
    if not os.path.exists(temp):
        print("\nuploads/temp.wav not found. Put the audio you used in UI at uploads/temp.wav and re-run.")
        return
    print("\nUsing file:", temp)

    # try user extractor
    user_out, reason = try_user_extractor(temp)
    print("User extractor result reason:", reason, " type:", type(user_out).__name__)
    if user_out is None:
        print("User extractor didn't return usable output. Using fallback extractor.")
        feature_dict = compute_features_fallback(temp, n_mfcc=20)
    else:
        # normalize user_out to dict
        if isinstance(user_out, dict):
            feature_dict = user_out
        else:
            # try numpy/array/series -> list -> map to feature_order if length matches
            try:
                arr = np.array(user_out).reshape(1,-1)
                if arr.shape[1] == len(feature_order):
                    feature_dict = {feature_order[i]: float(arr[0,i]) for i in range(len(feature_order))}
                else:
                    # maybe Series-like with index
                    try:
                        import pandas as pd
                        s = pd.Series(user_out)
                        feature_dict = s.to_dict()
                    except Exception:
                        # fallback: attempt to coerce named attributes
                        feature_dict = {}
                        print("Could not coerce user_out to dict; using fallback extractor instead.")
                        feature_dict = compute_features_fallback(temp, n_mfcc=20)
            except Exception:
                print("Failed to interpret user_out; using fallback extractor.")
                feature_dict = compute_features_fallback(temp, n_mfcc=20)

    # build vector and inspect values
    vec = build_vector_from_dict(feature_order, feature_dict)
    X_df = None
    try:
        import pandas as pd
        X_df = pd.DataFrame([vec], columns=feature_order)
    except Exception:
        X_df = vec.reshape(1,-1)

    # which indices non-zero
    vals = vec
    nz_idx = np.where(np.abs(vals) > 1e-9)[0]
    print(f"\nnonzero count: {len(nz_idx)} (showing up to first 80):")
    # show first 30 features with their values (compact)
    for i in range(min(len(feature_order),80)):
        if i < 20 or i in nz_idx[:60]:
            print(f" idx {i:2d} {feature_order[i]:25s} -> {vals[i]}")
    # show top important features and whether zero
    if hasattr(model, "feature_importances_"):
        order = np.argsort(model.feature_importances_)[::-1]
        print("\nTop 20 importances with actual vector values (zero? True/False):")
        for i in order[:20]:
            v = vals[i]
            print(f" {feature_order[i]:25s} -> imp:{model.feature_importances_[i]:0.4f}  val:{v:0.6f} zero? {abs(v) < 1e-6}")
    # scale & predict
    try:
        X_scaled = scaler.transform(X_df)
        pred_num = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
        if encoder is not None:
            try:
                label = encoder.inverse_transform([pred_num])[0]
            except Exception:
                label = pred_num
        else:
            label = pred_num
        print("\nPRED numeric:", pred_num, "label:", label)
        if proba is not None:
            print("Top probs (first 10):", np.round(proba[:10],3))
    except Exception as e:
        print("Scaling/predict failed:", e); traceback.print_exc()

    # final short mapping summary
    print("\n--- MAPPING CHECK ---")
    missing = []
    for name in feature_order:
        if name.lower() not in {k.lower() for k in feature_dict.keys()}:
            missing.append(name)
    print("feature_order entries not present in extractor output (showing up to 40):")
    for x in missing[:40]:
        print("  ", x)
    print("\nDone.")

if __name__ == "__main__":
    main()
