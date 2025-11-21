# inspect_mapping.py
import json, os, sys, re
import numpy as np
import librosa

ART = "artifacts"
FO = os.path.join(ART, "feature_order.json")      

def compute_features_fallback(path, n_mfcc=20, duration=3):
    y, sr = librosa.load(path, duration=duration, sr=None)    
    feats = {}     
    try:
        rms = librosa.feature.rms(y=y)[0]
        feats["rms_mean"] = float(np.mean(rms)); feats["rms_std"] = float(np.std(rms))
    except:
        feats["rms_mean"]=0.0; feats["rms_std"]=0.0
    try:
        z = librosa.feature.zero_crossing_rate(y)[0]
        feats["zero_crossing_rate_mean"]=float(np.mean(z)); feats["zero_crossing_rate_std"]=float(np.std(z))
    except:
        feats["zero_crossing_rate_mean"]=0.0; feats["zero_crossing_rate_std"]=0.0
    try:
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats["spectral_centroid_mean"]=float(np.mean(sc)); feats["spectral_centroid_std"]=float(np.std(sc))
    except:
        feats["spectral_centroid_mean"]=0.0; feats["spectral_centroid_std"]=0.0
    try:
        sbw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        feats["spectral_bandwidth_mean"]=float(np.mean(sbw)); feats["spectral_bandwidth_std"]=float(np.std(sbw))
    except:
        feats["spectral_bandwidth_mean"]=0.0; feats["spectral_bandwidth_std"]=0.0
    try:
        roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats["rolloff_mean"]=float(np.mean(roll)); feats["rolloff_std"]=float(np.std(roll))
    except:
        feats["rolloff_mean"]=0.0; feats["rolloff_std"]=0.0
    try:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        feats["chroma_mean"]=float(np.mean(chroma)); feats["chroma_std"]=float(np.std(chroma))
    except:
        feats["chroma_mean"]=0.0; feats["chroma_std"]=0.0
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo"] = float(np.asarray(tempo).ravel()[0]) if isinstance(tempo,(list,tuple,np.ndarray)) else float(tempo)
    except:
        feats["tempo"]=0.0
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mm = np.mean(mfcc.T, axis=0); ms = np.std(mfcc.T, axis=0)
        for i in range(len(mm)):
            feats[f"mfcc{i+1}_mean"]=float(mm[i]); feats[f"mfcc{i+1}_std"]=float(ms[i])
    except:
        for i in range(n_mfcc):
            feats[f"mfcc{i+1}_mean"]=0.0; feats[f"mfcc{i+1}_std"]=0.0
    # placeholders
    feats.setdefault("harmony_mean", 0.0); feats.setdefault("harmony_std", 0.0)
    feats.setdefault("percept_mean", 0.0); feats.setdefault("percept_std", 0.0)
    return feats

def normalize(k):
    k2 = k.strip().lower()
    k2 = re.sub(r"[^a-z0-9]+","_", k2)
    k2 = re.sub(r"_+","_", k2).strip("_")
    return k2

def variants(name):
    n = normalize(name)
    out = {n}
    # mfcc variants
    m = re.search(r"mfcc[_-]?(\d+).*?(mean|std|var)?", n)
    if m:
        idx = m.group(1)
        suff = m.group(2) or "mean"
        out.add(f"mfcc{idx}_{suff}"); out.add(f"mfcc_{idx}_{suff}")
        out.add(f"mfcc{idx}{suff}"); out.add(f"mfcc_{idx}{suff}")
        out.add(f"mfcc{idx}_m"); out.add(f"mfcc_{idx}_m")
    # mean/std/var synonyms
    out.add(n.replace("_mean","_avg")); out.add(n.replace("_std","_var"))
    out.add(n.replace("_var","_std"))
    out.add(n.replace("_",""))
    return out

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_mapping.py /path/to/sample.wav")
        sys.exit(1)
    wav = sys.argv[1]
    if not os.path.exists(FO):
        print("Missing:", FO); sys.exit(1)
    feature_order = json.load(open(FO))
    feats = compute_features_fallback(wav)
    norm_fd = {normalize(k): v for k,v in feats.items()}
    matched = []
    unmatched = []
    mapping = {}
    for name in feature_order:
        found = False
        for cand in variants(name):
            if cand in norm_fd:
                mapping[name] = cand
                matched.append(name)
                found = True
                break
        if not found:
            # substring fallback
            for k in norm_fd.keys():
                if k in normalize(name) or normalize(name) in k:
                    mapping[name] = k
                    matched.append(name)
                    found = True
                    break
        if not found:
            unmatched.append(name)
    print("=== STATS ===")
    print("feature_order_len:", len(feature_order))
    print("extractor_keys_count:", len(feats))
    print("matched_count:", len(matched))
    print("unmatched_count:", len(unmatched))
    print("\n=== sample extractor keys (first 30) ===")
    for i,k in enumerate(list(feats.keys())[:30]):
        print(f"{i+1:02d}. {k}")
    print("\n=== sample mapping (first 30) ===")
    n=0
    for k,v in list(mapping.items())[:30]:
        print(f"{k}  <-  {v}")
        n+=1
    print("\n=== first 30 unmatched feature_order names ===")
    for i,x in enumerate(unmatched[:30]):
        print(f"{i+1:02d}. {x}")
    print("\n\nIf many unmatched entries, fix by renaming feature_order.json keys or updating your extractor to produce the names shown above.")
