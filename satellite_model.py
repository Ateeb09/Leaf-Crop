import os
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error


DATASET_PATH = "satellite_dataset.csv"
N_SAMPLES = 10_000


def generate_synthetic_dataset(path: str, n_samples: int = N_SAMPLES) -> pd.DataFrame:
    """
    Synthetic "satellite/environment" dataset.
    Features: temperature, humidity, rainfall, pressure.
    Targets: soil_type (categorical), soil_moisture, wind_speed.
    """
    rng = np.random.default_rng(42)

    temperature = rng.normal(loc=28.0, scale=5.0, size=n_samples).clip(5, 45)
    humidity = rng.uniform(20, 100, size=n_samples)
    rainfall = rng.exponential(scale=5.0, size=n_samples).clip(0, 50)
    pressure = rng.normal(loc=1008.0, scale=8.0, size=n_samples).clip(980, 1040)

    # Simple synthetic relationships
    soil_moisture = (
        0.4 * (rainfall / 50.0) + 0.3 * (humidity / 100.0) - 0.1 * (temperature - 25) / 20.0
    )
    soil_moisture = np.clip(soil_moisture, 0.0, 1.0)

    wind_speed = rng.normal(
        loc=10.0 + (pressure - 1008.0) / 4.0 - (humidity - 60.0) / 15.0, scale=2.5, size=n_samples
    ).clip(0, 40)

    # Soil type buckets based on conditions
    soil_type = []
    for sm, rf in zip(soil_moisture, rainfall):
        if sm < 0.25 and rf < 5:
            soil_type.append("Sandy")
        elif sm > 0.6 or rf > 20:
            soil_type.append("Clay")
        else:
            soil_type.append("Loam")

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
            "pressure": pressure,
            "soil_moisture": soil_moisture,
            "wind_speed": wind_speed,
            "soil_type": soil_type,
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

    feature_cols = ["temperature", "humidity", "rainfall", "pressure"]
    X = df[feature_cols]

    # Soil type classifier
    y_soil = df["soil_type"]

    # Continuous targets (soil moisture, wind speed)
    target_cols_reg = ["soil_moisture", "wind_speed"]
    y_reg = df[target_cols_reg]

    X_train, X_test, y_soil_train, y_soil_test, y_reg_train, y_reg_test = train_test_split(
        X, y_soil, y_reg, test_size=0.2, random_state=42, stratify=y_soil
    )

    # Limit tree depth and min samples to reduce overfitting / memorization
    soil_clf = RandomForestClassifier(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=8,
        min_samples_split=15,
        random_state=42,
        n_jobs=-1,
    )
    soil_clf.fit(X_train, y_soil_train)
    soil_pred = soil_clf.predict(X_test)
    print("=== Soil type classifier report (test set) ===")
    print(classification_report(y_soil_test, soil_pred))

    cond_reg = RandomForestRegressor(
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=8,
        min_samples_split=15,
        random_state=42,
        n_jobs=-1,
    )
    cond_reg.fit(X_train, y_reg_train)
    y_reg_pred = cond_reg.predict(X_test)
    mse = mean_squared_error(y_reg_test, y_reg_pred)
    print(f"=== Condition regressor MSE: {mse:.4f} ===")

    bundle = {
        "feature_cols": feature_cols,
        "target_cols_reg": target_cols_reg,
        "soil_model": soil_clf,
        "cond_model": cond_reg,
    }

    joblib.dump(bundle, "satellite_model.pkl")

    meta = {
        "n_samples": int(len(df)),
        "feature_cols": feature_cols,
        "target_cols_reg": target_cols_reg,
        "soil_classes": sorted(df["soil_type"].unique().tolist()),
    }
    with open("satellite_model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("Satellite/environment model training complete.")


if __name__ == "__main__":
    main()

