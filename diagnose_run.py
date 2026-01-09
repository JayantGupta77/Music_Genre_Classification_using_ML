# diagnose_run.py
import joblib, importlib, os, json, traceback, librosa, numpy as np, pandas as pd, glob

print("cwd:", os.getcwd())

# load artifacts
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
try:
    encoder = joblib.load("artifacts/encoder.pkl")
except Exception:
    encoder = None
with open("artifacts/feature_order.json","r") as f:
    feature_order = json.load(f)

print("Loaded model, scaler, encoder present?", encoder is not None)
print("Feature order length:", len(feature_order))
if encoder is not None:
    try:
        print("Encoder classes:", list(getattr(encoder, "classes_", None)))
    except Exception:
        print("Encoder present but cannot show classes")

# try import model_utils
mu = None
try:
    mu = importlib.import_module("model_utils")
    print("model_utils public names (subset):", [n for n in dir(mu) if not n.startswith("_")][:80])
except Exception as e:
    print("model_utils import failed:", e)

# helper: try user extractor (generic)
def call_user_extractor(path):
    if mu is None:
        return None, "no_model_utils"
    names = [n for n in dir(mu) if "extract" in n.lower()]
    if not names:
        return None, "no_extract_names"
    for name in names:
        func = getattr(mu, name)
        try:
            out = func(path)
            return out, f"called:{name}(path)"
        except TypeError:
            try:
                y,sr = librosa.load(path, duration=3, sr=None)
                out = func(y,sr)
                return out, f"called:{name}(y,sr)"
            except Exception as e2:
                continue
        except Exception as e:
            continue
    return None, "all_candidates_failed"

# mapping function used in your app ( simplified copy)
import re
def build_vec_from_dict(feature_order, feature_dict):
    vec = np.zeros(len(feature_order), dtype=float)
    fd = {k.lower(): float(v) for k,v in feature_dict.items()}
    for i, name in enumerate(feature_order):
        n = name.lower()
        if n in fd:
            vec[i] = fd[n]; continue
        n2 = n.replace("-", "_").replace(" ", "_")
        if n2 in fd:
            vec[i] = fd[n2]; continue
        n2a = n2.replace("_","")
        if n2a in fd:
            vec[i] = fd[n2a]; continue
        m = re.search(r"(mfcc)[^\d]*(\d+)[^\w]*(mean|std|var)?", n)
        if m:
            base = f"mfcc{int(m.group(2))}_{'mean' if (m.group(3) is None or 'mean' in m.group(3)) else 'std'}"
            if base in fd:
                vec[i] = fd[base]; continue
        for k in fd.keys():
            if k in n:
                vec[i] = fd[k]; break
    return vec

# choose sample files from data/genres_original if available else uploads/temp.wav
candidates = glob.glob("data/genres_original/*/*.wav")
if not candidates:
    candidates = glob.glob("uploads/*.wav")
if not candidates:
    print("No sample wav files found in data/genres_original or uploads. Put some .wav files in uploads/")
    raise SystemExit(1)

# limit to 12 files for speed
candidates = candidates[:12]
print("Testing files:", candidates)

results = []
for p in candidates:
    try:
        print("\n--- FILE:", p)
        # 1) try user extractor
        out, reason = call_user_extractor(p)
        print("user extractor result reason:", reason, "type:", type(out))
        used_extractor = False
        vec = None
        if out is not None:
            # convert outputs
            import pandas as pd
            if isinstance(out, pd.Series):
                fd = out.to_dict()
                vec = build_vec_from_dict(feature_order, fd)
                used_extractor = True
                print("converted pandas Series -> feature dict, nonzero count:", np.count_nonzero(vec))
            elif isinstance(out, pd.DataFrame):
                fd = out.iloc[0].to_dict()
                vec = build_vec_from_dict(feature_order, fd)
                used_extractor = True
                print("converted pandas DataFrame -> feature dict, nonzero count:", np.count_nonzero(vec))
            elif isinstance(out, dict):
                vec = build_vec_from_dict(feature_order, out)
                used_extractor = True
                print("used dict from extractor, nonzero count:", np.count_nonzero(vec))
            else:
                arr = np.asarray(out).ravel()
                print("numeric array shape:", arr.shape)
                if arr.size == len(feature_order):
                    vec = arr.astype(float)
                    used_extractor = True
                    print("array length matches feature_order")
                else:
                    print("array length DOES NOT match feature_order:", arr.size)
        # if extractor not used, compute fallback features (same as app)
        if not used_extractor or vec is None:
            print("Using fallback features for file")
            y,sr = librosa.load(p, duration=3, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_std = np.std(mfcc.T, axis=0)
            featd = {"tempo": float(librosa.beat.beat_track(y=y, sr=sr)[0]) if True else 0.0}
            featd["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)[0]))
            for i in range(len(mfcc_mean)):
                featd[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
                featd[f"mfcc{i+1}_std"] = float(mfcc_std[i])
            vec = build_vec_from_dict(feature_order, featd)
            print("fallback vec nonzero count:", np.count_nonzero(vec))

        # build DataFrame, scale and predict
        X_df = pd.DataFrame([vec], columns=feature_order)
        print("feature slice first 20:", np.round(X_df.values[0][:20],4))
        X_scaled = scaler.transform(X_df)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None
        if encoder is not None:
            try:
                lab = encoder.inverse_transform([pred])[0]
            except Exception:
                try:
                    lab = encoder.inverse_transform([str(pred)])[0]
                except:
                    lab = pred
        else:
            lab = pred
        print("PRED numeric:", pred, "label:", lab)
        if proba is not None:
            print("proba slice:", np.round(proba[:8],3), "top:", np.max(proba))
        results.append((p, pred, lab, np.max(proba) if proba is not None else None, np.count_nonzero(vec)))
    except Exception as e:
        print("ERROR processing", p, e)
        traceback.print_exc()

# summary
preds = [r[1] for r in results]
print("\n=== SUMMARY ===")
print("files tested:", len(results))
print("unique numeric preds:", set(preds))
print("counts per numeric pred:")
from collections import Counter
print(Counter(preds))
