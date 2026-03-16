# check_encoder.py
import joblib
from collections import Counter

enc = None
try:
    enc = joblib.load("artifacts/encoder.pkl")
    print("Loaded encoder. classes_:", enc.classes_)
except Exception as e:
    print("No encoder or failed to load:", e)


# if model exists, show its overall predicted class distribution on saved training features(fast)
import pandas as pd, joblib
try:
    model = joblib.load("artifacts/model.pkl")
    feat = pd.read_csv("features_3_sec.csv")  # or features_30_sec.csv if you used 30s
    # assume last column is label named 'label' or 'genre' — adjust if needed
    possible_label_cols = [c for c in feat.columns if c.lower() in ("label","genre","class","target")]
    if possible_label_cols:
        y = feat[possible_label_cols[0]]
        X = feat.drop(columns=[possible_label_cols[0]])
    else:
        # if csv has feature columns only, try to load features and use model.predict to see distribution
        X = feat.copy()
        y = None

    # if scaler uses feature names, ensure columns align; keep only those matching model n_features_in_
    try:
        scaler = joblib.load("artifacts/scaler.pkl")
        # If scaler fitted with names, try using them
    except Exception:
        scaler = None

    # reduce to columns model expects if necessary
    # try predicting (may raise if columns mismatch)
    preds = model.predict(X)
    print("Model predicted classes example counts:", Counter(preds))
except Exception as e:
    print("Skipping bulk model test (needs features CSV aligned). Error:", e)
