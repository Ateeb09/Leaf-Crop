"""
Train the disease-risk model from weather features (temp, humidity, rainfall, pressure).
Uses regularized LogisticRegression to avoid memorization; falls back to synthetic data if CSV missing.
Run from project root. Output: risk_model.pkl (same interface as app expects: predict_proba(X)[0][1]).
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

CSV_PATH = "daily_weather_processed.csv"
FEATURE_COLS = ["temp", "humidity", "rainfall", "pressure"]


def load_or_synthetic():
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Map to names app uses: avg_temp, avg_humidity, total_rainfall; add pressure if missing
        if "avg_temp" in df.columns and "avg_humidity" in df.columns and "total_rainfall" in df.columns and "risk" in df.columns:
            X = df[["avg_temp", "avg_humidity", "total_rainfall"]].copy()
            X.columns = ["temp", "humidity", "rainfall"]
            if "pressure" not in df.columns:
                X["pressure"] = 1013.0  # default
            else:
                X["pressure"] = df["pressure"].values
            X = X[FEATURE_COLS]
            y = (df["risk"].values > 0.5).astype(int)  # binarize for classification
            return np.array(X, dtype=float), np.array(y)
    # Synthetic: risk increases with humidity and rainfall, decreases with pressure
    rng = np.random.default_rng(42)
    n = 5000
    temp = rng.normal(28, 6, n).clip(5, 45)
    humidity = rng.uniform(20, 100, n)
    rainfall = rng.exponential(5, n).clip(0, 50)
    pressure = rng.normal(1008, 10, n).clip(980, 1040)
    # Rule-based risk + noise (so model learns pattern, not exact rule)
    risk_cont = 0.3 * (humidity / 100.0) + 0.2 * (rainfall / 50.0) - 0.1 * (pressure - 1008) / 40
    risk_cont = np.clip(risk_cont + rng.normal(0, 0.15, n), 0, 1)
    y = (risk_cont > 0.5).astype(int)
    X = np.column_stack([temp, humidity, rainfall, pressure])
    return X, y


def main():
    X, y = load_or_synthetic()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Regularized classifier to avoid overfitting (C=0.5 = stronger L2)
    model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("=== Risk model (test set) ===")
    print(classification_report(y_test, y_pred, target_names=["low_risk", "high_risk"]))
    if len(np.unique(y_test)) > 1:
        print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

    joblib.dump(model, "risk_model.pkl")
    print("Saved risk_model.pkl (use with features: temp, humidity, rainfall, pressure).")


if __name__ == "__main__":
    main()
