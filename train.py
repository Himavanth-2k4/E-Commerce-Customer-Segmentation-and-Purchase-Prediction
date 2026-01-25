"""
Reproducible Training Pipeline for Customer Behavior & Satisfaction Prediction.
Loads raw dataset, performs feature encoding, standard scaling,
trains a Random Forest classifier, and evaluates performance.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


def train_model(data_path: str = None, output_dir: str = None):
    if data_path is None:
        data_path = os.path.join("backend", "E-commerce Customer Behavior - Sheet1.csv")
    if output_dir is None:
        output_dir = os.path.join("backend", "model")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    numerical_cols = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating', 'Days Since Last Purchase']
    categorical_cols = ['Gender', 'City', 'Membership Type']
    target_col = 'Satisfaction Level'

    # Filter out records missing the target
    df = df.dropna(subset=[target_col]).copy()

    # Encode categorical features
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Encode target
    y_le = LabelEncoder()
    y = y_le.fit_transform(df[target_col].astype(str))

    # Feature matrix in exact required feature order
    feature_order = ['Age', 'Total Spend', 'Items Purchased', 'Average Rating',
                     'Days Since Last Purchase', 'Gender', 'City', 'Membership Type']
    X = df[feature_order].copy()

    # Scaling
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in y_le.classes_]))

    # Export artifacts
    joblib.dump(clf, os.path.join(output_dir, "final_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(encoders['City'], os.path.join(output_dir, "city_le.pkl"))
    joblib.dump(encoders['Gender'], os.path.join(output_dir, "eng_le.pkl"))
    joblib.dump(encoders['Membership Type'], os.path.join(output_dir, "ship_le.pkl"))
    joblib.dump(y_le, os.path.join(output_dir, "y_le.pkl"))
    print(f"✓ All model artifacts successfully saved to {output_dir}")

    return acc


if __name__ == "__main__":
    train_model()
