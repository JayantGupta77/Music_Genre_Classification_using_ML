# train.py
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    data_path = os.path.join('data', 'features_3_sec.csv')
    artifacts_dir = 'artifacts'
    os.makedirs(artifacts_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print("Shape:", df.shape)
    drop_cols = [c for c in ['filename','length'] if c in df.columns]
    X = df.drop(columns=drop_cols + ['label'])
    y = df['label']

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Training RandomForest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(model, os.path.join(artifacts_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(artifacts_dir, 'scaler.pkl'))
    joblib.dump(encoder, os.path.join(artifacts_dir, 'encoder.pkl'))
    print("Saved artifacts to", artifacts_dir)

if __name__ == "__main__":
    main()

def main():
    # your full training logic is here...
    # example:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # model.fit(X_train, y_train)
    # predictions = model.predict(X_test)
    # ... (accuracy, report, etc.)
    # joblib.dump(model, os.path.join(artifacts_dir, "model.pkl"))

    # after training and saving model, now save the columns
    import joblib
    import os

    artifacts_dir = os.path.join(os.getcwd(), "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    # get column names from training DataFrame
    columns_list = X_train.columns.tolist()

    # save them
    joblib.dump(columns_list, os.path.join(artifacts_dir, "columns.pkl"))
    print("✅ columns.pkl saved successfully in artifacts folder.")

if __name__ == "__main__":
    main()
