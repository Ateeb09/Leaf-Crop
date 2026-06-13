import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


DATASET_PATH = "fertilizer_dataset.csv"
N_SAMPLES = 10_000


FERTILIZERS = [
    "Balanced NPK",
    "High Nitrogen",
    "High Phosphorus",
    "High Potassium",
    "Organic Compost",
]


def generate_synthetic_dataset(path: str, n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Synthetic fertilizer recommendation dataset.
    Features: crop, soil_type, temperature, humidity, rainfall, risk_index.
    Target: fertilizer label.
    """
    rng = np.random.default_rng(123)

    crops = [
        "Apple",
        "Bell Pepper",
        "Cherry",
        "Corn (Maize)",
        "Grape",
        "Peach",
        "Potato",
        "Strawberry",
        "Tomato",
    ]
    soil_types = ["Sandy", "Loam", "Clay"]

    crop = rng.choice(crops, size=n_samples)
    soil_type = rng.choice(soil_types, size=n_samples, p=[0.3, 0.4, 0.3])

    temperature = rng.normal(loc=28.0, scale=5.0, size=n_samples).clip(5, 45)
    humidity = rng.uniform(20, 100, size=n_samples)
    rainfall = rng.exponential(scale=5.0, size=n_samples).clip(0, 50)
    risk_index = rng.uniform(0, 1, size=n_samples)

    # Crop-sensitive rules so recommendations vary by crop as well as weather/soil
    def _pick_fertilizer(c, s, t, h, r, risk):
        if risk > 0.7:
            if s == "Sandy":
                return "Organic Compost"
            return "Balanced NPK"
        if t < 20 and h < 40:
            if c in ("Tomato", "Bell Pepper", "Strawberry"):
                return "High Nitrogen"
            if c in ("Corn (Maize)", "Potato"):
                return "High Phosphorus"
            return "High Nitrogen"
        if r < 3 and s == "Sandy":
            if c in ("Potato", "Tomato", "Grape"):
                return "High Potassium"
            return "High Potassium" if c in ("Strawberry", "Cherry") else "Balanced NPK"
        if s == "Clay":
            if c in ("Potato", "Corn (Maize)"):
                return "High Phosphorus"
            return "High Phosphorus"
        # Default by crop for variety
        if c in ("Tomato", "Bell Pepper"):
            return "Balanced NPK" if h < 70 else "High Potassium"
        if c in ("Potato", "Corn (Maize)"):
            return "High Phosphorus" if r > 10 else "Balanced NPK"
        if c in ("Apple", "Cherry", "Peach"):
            return "High Potassium" if t > 28 else "Balanced NPK"
        if c in ("Grape", "Strawberry"):
            return "Organic Compost" if risk > 0.5 else "Balanced NPK"
        return "Balanced NPK"

    fertilizers = [_pick_fertilizer(c, s, t, h, r, risk) for c, s, t, h, r, risk in zip(crop, soil_type, temperature, humidity, rainfall, risk_index)]

    df = pd.DataFrame(
        {
            "crop": crop,
            "soil_type": soil_type,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "risk_index": risk_index,
            "fertilizer": fertilizers,
        }
    )

    df.to_csv(path, index=False)
    return df


def main() -> None:
    # Load or create dataset
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        if len(df) > N_SAMPLES:
            df = df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)
    else:
        df = generate_synthetic_dataset(DATASET_PATH, N_SAMPLES)

    feature_cols = ["crop", "soil_type", "temperature", "humidity", "rainfall", "risk_index"]
    X = df[feature_cols]
    y = df["fertilizer"]

    # One-hot encode categorical features via pandas get_dummies
    X_enc = pd.get_dummies(X, columns=["crop", "soil_type"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, random_state=42, stratify=y
    )

    # Limit tree depth and min samples to reduce overfitting / memorization
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=8,
        min_samples_split=15,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("=== Fertilizer classifier report (test set) ===")
    print(classification_report(y_test, y_pred))

    bundle = {
        "model": clf,
        "feature_cols": feature_cols,
        "dummies_cols": list(X_enc.columns),
        "fertilizer_labels": sorted(df["fertilizer"].unique().tolist()),
    }

    joblib.dump(bundle, "fertilizer_model.pkl")

    meta = {
        "n_samples": int(len(df)),
        "feature_cols": feature_cols,
        "fertilizer_labels": sorted(df["fertilizer"].unique().tolist()),
    }
    with open("fertilizer_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Fertilizer recommendation model training complete.")


if __name__ == "__main__":
    main()

