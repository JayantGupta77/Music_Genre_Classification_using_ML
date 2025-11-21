# show_model_info.py
import joblib
import numpy as np
import json

model = joblib.load("artifacts/model.pkl")
print("Model type:", type(model))
# classes_ attribute may exist on the model (some classifiers) or only in encoder
print("Model has attribute 'classes_'?:", hasattr(model, "classes_"))
if hasattr(model, "classes_"):
    print("model.classes_:", model.classes_)

try:
    enc = joblib.load("artifacts/encoder.pkl")
    print("Loaded encoder type:", type(enc))
    # many encoders store classes_ (LabelEncoder)
    print("encoder.classes_:", getattr(enc, "classes_", None))
except Exception as e:
    print("Could not load encoder.pkl:", e)
    enc = None

# if model supports predict_proba, show class indices -> shape
if hasattr(model, "predict_proba"):
    # print a fake all-zero sample with right dimension just to inspect proba length
    try:
        n_classes = len(model.classes_) if hasattr(model, "classes_") else None
        print("Model supports predict_proba. reported n_classes (model.classes_):", n_classes)
    except Exception:
        pass
    print("predict_proba available: True")
else:
    print("predict_proba available: False")

# show feature-order length (if present)
try:
    with open("artifacts/feature_order.json","r") as f:
        fo = json.load(f)
    print("feature_order length:", len(fo))
except Exception as e:
    print("Could not load feature_order.json:", e)
