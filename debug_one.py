# debug_one.py — run with: python debug_one.py
import joblib, json, os, numpy as np, pandas as pd, librosa, re
from importlib import import_module


# debug_predict.py
import joblib, json, numpy as np, pandas as pd, os, librosa
from pprint import pprint

ART = "artifacts"
MODEL_PATH = os.path.join(ART, "model.pkl")
SCALER_PATH = os.path.join(ART, "scaler.pkl")
ENC_PATH = os.path.join(ART, "encoder.pkl")
FEATURE_ORDER = os.path.join(ART, "feature_order.json")

def build_feature_vector_from_dict(feature_order, feature_dict):
    # copy the same mapping code you use in app.py (simple fuzzy)
    import re
    vec = np.zeros(len(feature_order), dtype=float)
    fd = {k.lower(): float(v) for k, v in feature_dict.items()}
    for i, name in enumerate(feature_order):
        n = name.lower()
        if n in fd:
            vec[i] = fd[n]; continue
        n2 = n.replace("-", "_").replace(" ", "_")
        if n2 in fd:
            vec[i] = fd[n2]; continue
        n2a = n2.replace("_", "")
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

def compute_features_fallback(path, n_mfcc=20, duration=3):
    y, sr = librosa.load(path, duration=duration, sr=None)
    feats = {}
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms))
    except: feats["rms_mean"] = 0.0
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, (list, tuple, np.ndarray)):
            tempo = float(np.asarray(tempo).ravel()[0])
        feats["tempo"] = float(tempo)
    except: feats["tempo"] = 0.0
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mmeans = np.mean(mfcc.T, axis=0)
        for i,v in enumerate(mmeans):
            feats[f"mfcc{i+1}_mean"] = float(v)
    except:
        for i in range(n_mfcc):
            feats[f"mfcc{i+1}_mean"] = 0.0
    return feats

def run_debug(audio_path):
    print("\n\n===== DEBUG for:", audio_path, "=====\n")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    try:
        enc = joblib.load(ENC_PATH)
        print("Loaded encoder.classes_:", getattr(enc, "classes_", None))
    except Exception as e:
        enc = None
        print("No encoder:", e)

    with open(FEATURE_ORDER,"r") as f:
        feature_order = json.load(f)
    print("feature_order length:", len(feature_order))
    print("model.classes_:", getattr(model, "classes_", None))
    print("model supports predict_proba?:", hasattr(model, "predict_proba"))

    # 1) try user extractor from model_utils if exists
    user_out = None
    try:
        import importlib
        m = importlib.import_module("model_utils")
        cand = [n for n in dir(m) if "extract" in n.lower()]
        print("model_utils extractors found:", cand)
        for name in cand:
            try:
                func = getattr(m, name)
                print("Calling", name, "with file path...")
                user_out = func(audio_path)
                print("-> returned type:", type(user_out))
                break
            except TypeError:
                try:
                    y, sr = librosa.load(audio_path, duration=3, sr=None)
                    user_out = func(y, sr); print("-> returned type:", type(user_out)); break
                except Exception as ee:
                    print("-> call failed:", ee)
                    continue
            except Exception as e:
                print("-> call failed:", e)
                continue
    except Exception as e:
        print("No model_utils or failed to import:", e)

    # 2) if user_out is dict-like use it, else fallback
    if user_out is None:
        print("No user extractor output — using fallback extractor")
        feat_dict = compute_features_fallback(audio_path)
    else:
        if isinstance(user_out, dict):
            feat_dict = user_out
        elif hasattr(user_out, "to_dict"):
            feat_dict = user_out.to_dict()
        else:
            # try to convert array->1D vector keyed by feature_order length
            try:
                arr = np.asarray(user_out).ravel()
                if arr.size == len(feature_order):
                    feat_dict = {feature_order[i]: float(arr[i]) for i in range(len(feature_order))}
                else:
                    print("user_out array size mismatch -> using fallback")
                    feat_dict = compute_features_fallback(audio_path)
            except Exception as e:
                print("cannot decode user_out -> fallback", e)
                feat_dict = compute_features_fallback(audio_path)

    # 3) build vector
    vec = build_feature_vector_from_dict(feature_order, feat_dict)
    X_df = pd.DataFrame([vec], columns=feature_order)
    print("\nNON-ZERO feature count:", np.count_nonzero(np.abs(vec) > 1e-9))
    nz_idx = np.where(np.abs(vec) > 1e-9)[0]
    print("Nonzero indices (first 40):", nz_idx[:40].tolist())
    print("Nonzero names/values (first 40):")
    for i in nz_idx[:40]:
        print(i, feature_order[i], "->", vec[i])

    print("\nFIRST 20 raw slice:", vec[:20].tolist())

    # 4) scaled
    Xs = scaler.transform(X_df)
    print("\nFIRST 20 scaled slice:", np.round(Xs[0][:20],6).tolist())

    # 5) predict & probs
    pred = model.predict(Xs)[0]
    print("\nModel numeric pred:", pred)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(Xs)[0]
        topk = sorted(enumerate(probs), key=lambda x: -x[1])[:6]
        print("Top probs (index->prob):", topk)
    if enc is not None:
        try:
            label = enc.inverse_transform([pred])[0]
        except Exception:
            label = pred
    else:
        label = pred
    print("Decoded label:", label)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python debug_predict.py <path_to_wav>")
        sys.exit(1)
    run_debug(sys.argv[1])


ART = "artifacts"
MODEL_PKL = os.path.join(ART, "model.pkl")
SCALER_PKL = os.path.join(ART, "scaler.pkl")
ENC_PKL = os.path.join(ART, "encoder.pkl")
FEATURE_ORDER = os.path.join(ART, "feature_order.json")

# load artifacts
model = joblib.load(MODEL_PKL)
scaler = joblib.load(SCALER_PKL)
try:
    encoder = joblib.load(ENC_PKL)
except Exception:
    encoder = None

with open(FEATURE_ORDER, "r") as f:
    feature_order = json.load(f)

# try to use model_utils extractor if present
def try_user_extractor(path):
    try:
        m = import_module("model_utils")
        for fn in ("extract_features", "make_features", "get_features",
                   "compute_features", "extract", "extract_features_from_file"):
            if hasattr(m, fn):
                func = getattr(m, fn)
                try:
                    return func(path)
                except TypeError:
                    try:
                        y, sr = librosa.load(path, duration=3, sr=None)
                        return func(y, sr)
                    except Exception:
                        continue
        return None
    except Exception:
        return None

def build_vector_from_dict(feature_dict):
    # simple mapping: try direct lowercase keys and a few normalisations
    fd = {k.lower(): float(v) for k, v in (feature_dict.items() if hasattr(feature_dict, "items") else {})}
    vec = np.zeros(len(feature_order), dtype=float)
    for i, name in enumerate(feature_order):
        n = name.lower()
        if n in fd:
            vec[i] = fd[n]; continue
        n2 = n.replace("-", "_").replace(" ", "_")
        if n2 in fd:
            vec[i] = fd[n2]; continue
        n2a = n2.replace("_", "")
        if n2a in fd:
            vec[i] = fd[n2a]; continue
        m = re.search(r"(mfcc)[^\d]*(\d+)[^\w]*(mean|std|var)?", n)
        if m:
            base = f"mfcc{int(m.group(2))}_{'mean' if (m.group(3) is None or 'mean' in m.group(3)) else 'std'}"
            if base in fd:
                vec[i] = fd[base]; continue
        # substring fallback
        for k in fd.keys():
            if k in n:
                vec[i] = fd[k]; break
    return vec

# path to inspect (same file the app saves)
path = "uploads/temp.wav"
if not os.path.exists(path):
    print("ERROR: file not found:", path)
    raise SystemExit(1)

print("Using file:", path)

# 1) try user extractor
user_out = try_user_extractor(path)
if user_out is not None:
    print("User extractor returned type:", type(user_out))
    # if it's array-like of same length as feature_order, use directly
    try:
        arr = np.array(user_out).reshape(1, -1)
        if arr.shape[1] == len(feature_order):
            X_df = pd.DataFrame(arr, columns=feature_order)
        else:
            # try dict conversion
            try:
                feature_dict = dict(user_out)
            except Exception:
                feature_dict = {}
            vec = build_vector_from_dict(feature_dict)
            X_df = pd.DataFrame([vec], columns=feature_order)
    except Exception:
        try:
            feature_dict = dict(user_out)
        except Exception:
            feature_dict = {}
        vec = build_vector_from_dict(feature_dict)
        X_df = pd.DataFrame([vec], columns=feature_order)
    print("Used user extractor.")
else:
    # fallback simple MFCC mean/std (keeps minimal)
    print("No user extractor — computing fallback MFCC mean/std.")
    y, sr = librosa.load(path, duration=3, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std  = np.std(mfcc.T, axis=0)
    feat = {}
    for i in range(len(mfcc_mean)):
        feat[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feat[f"mfcc{i+1}_std"]  = float(mfcc_std[i])
    vec = build_vector_from_dict(feat)
    X_df = pd.DataFrame([vec], columns=feature_order)

# show which columns are non-zero
vals = X_df.values[0]
nz_idx = np.where(np.abs(vals) > 1e-9)[0]
print(f"nonzero count: {len(nz_idx)} (showing up to first 80):")
for i in nz_idx[:80]:
    print(f"  idx {i:02d}  {feature_order[i]:30s} -> {vals[i]: .6f}")

print("\nFirst 20 raw slice:", np.round(vals[:20], 6).tolist())

# scale & predict (so we reproduce model behaviour)
X_scaled = scaler.transform(X_df)
pred = model.predict(X_scaled)[0]
print("\nModel numeric pred:", pred)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_scaled)[0]
    print("Top probs (first 10):", [round(float(x),3) for x in proba[:10]])
if encoder is not None:
    try:
        label = encoder.inverse_transform([pred])[0]
    except Exception:
        label = pred
else:
    label = pred
print("Decoded label:", label)
