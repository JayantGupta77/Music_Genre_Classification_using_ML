# show_importances.py
import joblib, json, numpy as np, pandas as pd

model = joblib.load("artifacts/model.pkl")
with open("artifacts/feature_order.json","r") as f:
    feature_order = json.load(f)

print("Model type:", type(model))
# get feature importances if available
if hasattr(model, "feature_importances_"):
    imp = np.array(model.feature_importances_)
    # show top 20 features by importance
    idx = np.argsort(imp)[::-1]
    print("Top 20 feature importances (index -> name -> importance):")
    for i in idx[:20]:
        name = feature_order[i] if i < len(feature_order) else "(out of range)"
        print(f"  idx {i:2d} -> {name:25s} -> {imp[i]:.4f}")
    # also list any very small ones
    zero_count = np.sum(imp <= 1e-6)
    print(f"\nFeatures with near-zero importance: {zero_count} / {len(imp)}")
else:
    print("Model has no feature_importances_ (not a tree ensemble?)")

# quick sanity: show how many features in feature_order and first 20 names
print("\nfeature_order length:", len(feature_order))
print("feature_order first 20:", feature_order[:20])
