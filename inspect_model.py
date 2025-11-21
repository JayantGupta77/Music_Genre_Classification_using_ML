# inspect_model.py
import os, joblib, json, numpy as np
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
ART = BASE / "artifacts"
MODEL_P = ART / "model.pkl"
ENC_P = ART / "encoder.pkl"
FEATURE_ORDER_P = ART / "feature_order.json"
SAMPLE = BASE / "uploads" / "temp.wav"   # file your app saves on upload

print("=== Paths ===")
print("model:", MODEL_P.exists(), MODEL_P)
print("encoder:", ENC_P.exists(), ENC_P)
print("feature_order:", FEATURE_ORDER_P.exists(), FEATURE_ORDER_P)
print("sample exists:", SAMPLE.exists(), SAMPLE)

# load model & optional encoder
model = joblib.load(str(MODEL_P))
enc = None
if ENC_P.exists():
    try:
        enc = joblib.load(str(ENC_P))
    except Exception as e:
        print("Failed to load encoder:", e)

# show model class info (if available)
print("\n=== Model class attributes ===")
print("hasattr(model,'classes_'):", hasattr(model, "classes_"))
if hasattr(model, "classes_"):
    print("model.classes_ (len):", len(getattr(model, "classes_")), getattr(model, "classes_")[:50])
else:
    print("model has no classes_ attribute. Type:", type(model))

# show encoder classes
print("\n=== Encoder info ===")
if enc is not None:
    try:
        classes = getattr(enc, "classes_", None)
        print("encoder type:", type(enc))
        print("encoder.classes_ exists:", classes is not None)
        if classes is not None:
            print("encoder.classes_ (len):", len(classes), classes[:50])
    except Exception as e:
        print("Error inspecting encoder:", e)
else:
    print("encoder not present.")

# show feature order length
if FEATURE_ORDER_P.exists():
    fo = json.load(open(FEATURE_ORDER_P))
    print("\nfeature_order length:", len(fo))
    print("first 20 feature_order:", fo[:20])
else:
    fo = None

# If sample exists, attempt a single predict_proba/predict to see numeric label and mapping
if SAMPLE.exists():
    print("\n=== Running sample predict on", SAMPLE)
    # we need the scaler and the same vector building as your app uses -> we will call your app's scaler and feature_order
    scaler_p = ART / "scaler.pkl"
    if not scaler_p.exists():
        print("scaler.pkl not found at", scaler_p)
    else:
        scaler = joblib.load(str(scaler_p))
        # try to reuse model_utils.extractor if exists to build features (best-effort)
        try:
            import model_utils
            feat_dict = model_utils.extract_features_from_file(str(SAMPLE))
            print("Got feat_dict from model_utils.extract_features_from_file (keys sample):", list(feat_dict.keys())[:30])
        except Exception as e:
            print("model_utils failed / not available:", e)
            # fallback: cannot build vector here reliably; exit
            print("-> Please call your /debug_features endpoint to get vector details (or run the inspect_model.py after building vector).")
            feat_dict = None

        if feat_dict is not None and fo is not None:
            # align vector like app does
            vec = []
            missing = []
            for col in fo:
                if col in feat_dict:
                    vec.append(float(feat_dict[col]))
                else:
                    vec.append(0.0); missing.append(col)
            X = np.array(vec).reshape(1, -1)
            Xs = scaler.transform(pd.DataFrame(X, columns=fo))
            pred = model.predict(Xs)
            print("model.predict ->", pred, " type:", type(pred), " value:", pred[0])
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(Xs)[0]
                    print("predict_proba (slice):", np.round(probs[:10], 3).tolist())
                except Exception as e:
                    print("predict_proba failed:", e)
            # if encoder present, show mapping attempts
            if enc is not None:
                try:
                    mapped = enc.inverse_transform(pred)
                    print("encoder.inverse_transform ->", mapped)
                except Exception as e:
                    print("encoder.inverse_transform failed:", e)
            else:
                # if model.classes_ exists, print mapping
                if hasattr(model, "classes_"):
                    try:
                        classes = model.classes_
                        idx = int(pred[0]) if np.issubdtype(type(pred[0]), np.integer) else None
                        print("model.classes_ mapping:", classes)
                        if idx is not None and 0 <= idx < len(classes):
                            print("mapped via model.classes_[pred]:", classes[idx])
                    except Exception as e:
                        print("class mapping check failed:", e)
else:
    print("\nNo sample file to run predict. Upload a file via the UI so app saves uploads/temp.wav, then run this script again.")
