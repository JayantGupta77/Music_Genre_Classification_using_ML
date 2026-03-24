# compare_extractors.py
import joblib, json, os, traceback
import numpy as np, pandas as pd, librosa, re
def build_feature_vector_from_dict(feature_order, feature_dict):
    vec = np.zeros(len(feature_order), dtype=float)
    fd = {k.lower(): float(v) for k, v in feature_dict.items()}
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


def compute_features_fallback(path, n_mfcc=20, duration=3):
    y, sr = librosa.load(path, duration=duration, sr=None)
    f = {}
    try:
        rms = librosa.feature.rms(y=y)[0]
        f["rms_mean"] = float(np.mean(rms)); f["rms_std"] = float(np.std(rms))
    except: f["rms_mean"]=0.0; f["rms_std"]=0.0
    try:
        z = librosa.feature.zero_crossing_rate(y)[0]
        f["zero_crossing_rate_mean"]=float(np.mean(z)); f["zero_crossing_rate_std"]=float(np.std(z))
    except: f["zero_crossing_rate_mean"]=0.0; f["zero_crossing_rate_std"]=0.0
    try:
        sc = librosa.feature.spectral_centroid(y=y,sr=sr)[0]
        f["spectral_centroid_mean"]=float(np.mean(sc)); f["spectral_centroid_std"]=float(np.std(sc))
    except: f["spectral_centroid_mean"]=0.0; f["spectral_centroid_std"]=0.0
    try:
        sbw = librosa.feature.spectral_bandwidth(y=y,sr=sr)[0]
        f["spectral_bandwidth_mean"]=float(np.mean(sbw)); f["spectral_bandwidth_std"]=float(np.std(sbw))
    except: f["spectral_bandwidth_mean"]=0.0; f["spectral_bandwidth_std"]=0.0
    try:
        roll = librosa.feature.spectral_rolloff(y=y,sr=sr)[0]
        f["rolloff_mean"]=float(np.mean(roll)); f["rolloff_std"]=float(np.std(roll))
    except: f["rolloff_mean"]=0.0; f["rolloff_std"]=0.0
    try:
        chroma = librosa.feature.chroma_stft(y=y,sr=sr)
        f["chroma_mean"]=float(np.mean(chroma)); f["chroma_std"]=float(np.std(chroma))
    except: f["chroma_mean"]=0.0; f["chroma_std"]=0.0
    try:
        tempo,_ = librosa.beat.beat_track(y=y,sr=sr)
        if isinstance(tempo,(list,tuple,np.ndarray)): tempo=float(np.asarray(tempo).ravel()[0])
        f["tempo"]=float(tempo)
    except: f["tempo"]=0.0
    try:
        mfcc = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
        mm = np.mean(mfcc.T,axis=0); ms = np.std(mfcc.T,axis=0)
        for i in range(len(mm)):
            f[f"mfcc{i+1}_mean"]=float(mm[i]); f[f"mfcc{i+1}_std"]=float(ms[i])
    except:
        for i in range(n_mfcc):
            f[f"mfcc{i+1}_mean"]=0.0; f[f"mfcc{i+1}_std"]=0.0
    f.setdefault("harmony_mean",0.0); f.setdefault("harmony_std",0.0)
    return f

# load artifacts
art_dir = "artifacts"
model = joblib.load(os.path.join(art_dir,"model.pkl"))
scaler = joblib.load(os.path.join(art_dir,"scaler.pkl"))
enc = None
try:
    enc = joblib.load(os.path.join(art_dir,"encoder.pkl"))
except: enc=None
with open(os.path.join(art_dir,"feature_order.json")) as fh:
    feature_order = json.load(fh)

audio = "uploads/temp.wav"
if not os.path.exists(audio):
    print("uploads/temp.wav not found — upload one from your browser first and hit Predict (so file exists).")
    raise SystemExit

# 1) try user's extractor
user_out = None
try:
    import importlib
    mu = importlib.import_module("model_utils")
    cand = [n for n in dir(mu) if "extract" in n.lower()]
    print("model_utils extractors found:", cand)
    for name in cand:
        func = getattr(mu,name)
        try:
            user_out = func(audio)
            print("Called", name, "-> returned type", type(user_out))
            break
        except TypeError:
            try:
                y,sr = librosa.load(audio,duration=3,sr=None)
                user_out = func(y,sr)
                print("Called", name, "(y,sr) -> returned type", type(user_out))
                break
            except Exception as e:
                print("func",name,"call failed:", e)
        except Exception as e:
            print("func",name,"call failed:", e)
except Exception as e:
    print("No model_utils or import error:", e)

def to_vec_from_user_out(user_out):
    if user_out is None: return None, {}
    if isinstance(user_out,pd.Series):
        return build_feature_vector_from_dict(feature_order, user_out.to_dict()), user_out.to_dict()
    if isinstance(user_out,pd.DataFrame):
        return build_feature_vector_from_dict(feature_order, user_out.iloc[0].to_dict()), user_out.iloc[0].to_dict()
    if isinstance(user_out,dict):
        return build_feature_vector_from_dict(feature_order, user_out), user_out
    try:
        arr = np.asarray(user_out).ravel()
        if arr.size == len(feature_order):
            return arr.astype(float), {}
        # else try map-like
        try:
            d = dict(user_out)
            return build_feature_vector_from_dict(feature_order, d), d
        except Exception:
            return None, {}
    except Exception:
        return None, {}

vec_user, dict_user = to_vec_from_user_out(user_out)
vec_fallback = None
dict_fallback = compute_features_fallback(audio)
vec_fallback = build_feature_vector_from_dict(feature_order, dict_fallback)

def report(name, vec, dict_repr):
    print("\n---",name,"---")
    print("nonzero count:", int(np.count_nonzero(np.abs(vec)>1e-9)),"/",len(vec))
    print("first 20 raw slice:", np.round(vec[:20],4))
    df = pd.DataFrame([vec], columns=feature_order)
    Xs = scaler.transform(df)
    print("first 20 scaled slice:", np.round(Xs[0,:20],4))
    pred = model.predict(Xs)[0]
    proba = model.predict_proba(Xs)[0] if hasattr(model,"predict_proba") else None
    label = pred
    if enc is not None:
        try:
            label = enc.inverse_transform([pred])[0]
        except Exception:
            pass
    print("prediction numeric:", pred, "label:", label)
    if proba is not None:
        print("top probs (slice):", np.round(proba[:10],3))
    # show top nonzero features
    nz = np.where(np.abs(vec)>1e-9)[0]
    print("some nonzero feature indexes (first 30):", nz[:30])
    if dict_repr:
        print("sample of user dict keys (first 40):", list(dict_repr.keys())[:40])

print("Running compare on:",audio)
report("USER-EXTRACTOR", vec_user if vec_user is not None else np.zeros(len(feature_order)), dict_user)
report("FALLBACK", vec_fallback, dict_fallback)
