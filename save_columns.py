# save_columns.py
import pandas as pd, json, os
df = pd.read_csv('data/features_3_sec.csv')
drop = [c for c in ['filename','length','label'] if c in df.columns]
cols = [c for c in df.columns if c not in drop]
os.makedirs('artifacts', exist_ok=True)
with open('artifacts/feature_order.json','w') as f:
    json.dump(cols,f)
print("Saved artifacts/feature_order.json with", len(cols), "columns.")
