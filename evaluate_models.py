# evaluate_model.py
import joblib, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
enc = None
try:
    enc = joblib.load("artifacts/encoder.pkl")
except:
    enc = None

df = pd.read_csv("features_3_sec.csv")  # change path if needed
# find label column
label_col = None
for c in df.columns:
    if c.lower() in ("label","genre","class","target"):
        label_col = c; break
if label_col is None:
    raise SystemExit("No label column found in CSV; open the file and tell me which column is the true label")

X = df.drop(columns=[label_col])
y = df[label_col]
# If scaler expects specific columns, try to reindex X
try:
    Xs = scaler.transform(X)
except Exception:
    # try using only columns that scaler/model expected
    print("Scaler transform failed on CSV features. Trying to align columns using feature_order.json")
    import json
    with open("artifacts/feature_order.json","r") as f: feature_order = json.load(f)
    X = X.reindex(columns=feature_order, fill_value=0)
    Xs = scaler.transform(X)

pred = model.predict(Xs)
if enc is not None:
    try:
        y_labels = enc.inverse_transform(y)
        pred_labels = enc.inverse_transform(pred)
    except Exception:
        y_labels = y; pred_labels = pred
else:
    y_labels = y; pred_labels = pred

print(classification_report(y_labels, pred_labels))
print("Confusion matrix:")
print(confusion_matrix(y_labels, pred_labels))
