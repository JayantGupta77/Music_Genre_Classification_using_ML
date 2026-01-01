# diagnose_inputs.py
import glob, os, json, joblib, numpy as np, pandas as pd, librosa, re
from collections import defaultdict

with open("artifacts/feature_order.json","r") as f:
    feature_order = json.load(f)
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
try:
    encoder = joblib.load("artifacts/encoder.pkl")
except:
    encoder = None

def build_basic_features(path):
    y, sr = librosa.load(path, duration=3, sr=None)
    feats = {}
    # MFCC mean/std
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    mfcc_std = np.std(mfcc.T, axis=0)
    for i in range(len(mfcc_mean)):
        feats[f"mfcc{i+1}_mean"] = float(mfcc_mean[i])
        feats[f"mfcc{i+1}_std"] = float(mfcc_std[i])
    # a handful of other features
    try:
        feats["tempo"] = float(librosa.beat.beat_track(y=y, sr=sr)[0])
    except:
        feats["tempo"] = 0.0
    try:
        feats["rms_mean"] = float(np.mean(librosa.feature.rms(y=y)[0]))
    except:
        feats["rms_mean"] = 0.0
    return feats

def build_vector(feat_dict):
    vec = np.zeros(len(feature_order))
    fd = {k.lower(): float(v) for k,v in feat_dict.items()}
    for i,name in enumerate(feature_order):
        n = name.lower()
        if n in fd:
            vec[i] = fd[n]; continue
        # try match 'mfcc' patterns
        m = re.search(r"(\d+)", n)
        if m and "mfcc" in n:
            idx = int(m.group(1))
            key_mean = f"mfcc{idx}_mean"
            key_std = f"mfcc{idx}_std"
            if key_mean in fd and "mean" in n: vec[i] = fd[key_mean]
            elif key_std in fd and ("std" in n or "var" in n): vec[i] = fd[key_std]
    return vec

# collect sample files (change path if needed)
files = glob.glob("data/**/*.wav", recursive=True)[:40]  # first 40 files
summary = defaultdict(list)

for p in files:
    feats = build_basic_features(p)
    vec = build_vector(feats)
    nonzeros = np.count_nonzero(vec)
    X_df = pd.DataFrame([vec], columns=feature_order)
    try:
        Xs = scaler.transform(X_df)
        pred = model.predict(Xs)[0]
        proba = model.predict_proba(Xs)[0] if hasattr(model,"predict_proba") else None
    except Exception as e:
        pred = f"error:{e}"
        proba = None
    label = encoder.inverse_transform([pred])[0] if encoder is not None and not isinstance(pred,str) else pred
    print(f"FILE: {os.path.basename(p):30s} nonzero_features:{nonzeros:2d} pred:{pred} label:{label} conf:{(proba[int(pred)] if proba is not None and isinstance(pred,(int,np.integer)) else 'NA')}")
    summary['nonzeros'].append(nonzeros)
    summary['preds'].append(pred)

import statistics
print("SUMMARY: avg nonzero=", statistics.mean(summary['nonzeros']) if summary['nonzeros'] else 0, "unique preds:", set(summary['preds']))
