"""
Step 3: Build real satellite and fertilizer datasets from your weather data.
Run this once, then run satellite_model.py and fertilizer_model.py to retrain.
"""
import numpy as np
import pandas as pd

N_SAMPLES = 10_000
CROPS = [
    "Apple", "Bell Pepper", "Cherry", "Corn (Maize)", "Grape",
    "Peach", "Potato", "Strawberry", "Tomato",
]


def _soil_moisture(rainfall: np.ndarray, humidity: np.ndarray, temp: np.ndarray) -> np.ndarray:
    sm = 0.4 * (rainfall / 50.0) + 0.3 * (humidity / 100.0) - 0.1 * (temp - 25) / 20.0
    return np.clip(sm, 0.0, 1.0)


def _soil_type(soil_moisture: np.ndarray, rainfall: np.ndarray) -> list:
    out = []
    for sm, rf in zip(soil_moisture, rainfall):
        if sm < 0.25 and rf < 5:
            out.append("Sandy")
        elif sm > 0.6 or rf > 20:
            out.append("Clay")
        else:
            out.append("Loam")
    return out


def _fertilizer_rule(crop: str, soil_type: str, temp: float, humidity: float, rainfall: float, risk: float) -> str:
    if risk > 0.7:
        return "Organic Compost" if soil_type == "Sandy" else "Balanced NPK"
    if temp < 20 and humidity < 40:
        return "High Nitrogen"
    if rainfall < 3 and soil_type == "Sandy":
        return "High Potassium" if crop in ("Potato", "Tomato", "Grape", "Strawberry", "Cherry") else "Balanced NPK"
    if soil_type == "Clay":
        return "High Phosphorus"
    # Vary by crop so model learns different recommendations
    if crop in ("Tomato", "Bell Pepper") and humidity >= 70:
        return "High Potassium"
    if crop in ("Potato", "Corn (Maize)") and rainfall > 10:
        return "High Phosphorus"
    if crop in ("Apple", "Cherry", "Peach") and temp > 28:
        return "High Potassium"
    if crop in ("Grape", "Strawberry") and risk > 0.5:
        return "Organic Compost"
    return "Balanced NPK"


def build_satellite_from_bengaluru():
    df = pd.read_csv("bengaluru.csv")
    # Map columns: tempC, humidity, precipMM, pressure, windspeedKmph
    temperature = df["tempC"].values
    humidity = df["humidity"].values.astype(float)
    rainfall = df["precipMM"].values.astype(float)
    pressure = df["pressure"].values.astype(float)
    wind_speed = df["windspeedKmph"].values.astype(float)

    soil_moisture = _soil_moisture(rainfall, humidity, temperature)
    soil_type = _soil_type(soil_moisture, rainfall)

    out = pd.DataFrame({
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "pressure": pressure,
        "soil_moisture": soil_moisture,
        "wind_speed": wind_speed,
        "soil_type": soil_type,
    })
    out = out.sample(n=min(N_SAMPLES, len(out)), random_state=42).reset_index(drop=True)
    out.to_csv("satellite_dataset.csv", index=False)
    print(f"Saved real satellite_dataset.csv with {len(out)} rows from bengaluru.csv")
    return out


def build_fertilizer_from_daily_weather():
    df = pd.read_csv("daily_weather_processed.csv")
    temperature = df["avg_temp"].values
    humidity = df["avg_humidity"].values
    rainfall = df["total_rainfall"].values
    risk = df["risk"].values

    soil_moisture = _soil_moisture(rainfall, humidity, temperature)
    soil_type = _soil_type(soil_moisture, rainfall)

    # Replicate rows with each crop to get enough variety, then sample 10k
    rows = []
    for i in range(len(df)):
        for crop in CROPS:
            rows.append({
                "crop": crop,
                "soil_type": soil_type[i],
                "temperature": temperature[i],
                "humidity": humidity[i],
                "rainfall": rainfall[i],
                "risk_index": float(risk[i]),
            })
    full = pd.DataFrame(rows)
    full = full.sample(n=min(N_SAMPLES, len(full)), random_state=42).reset_index(drop=True)

    fertilizers = [
        _fertilizer_rule(
            row["crop"], row["soil_type"], row["temperature"],
            row["humidity"], row["rainfall"], row["risk_index"]
        )
        for _, row in full.iterrows()
    ]
    full["fertilizer"] = fertilizers
    full.to_csv("fertilizer_dataset.csv", index=False)
    print(f"Saved real fertilizer_dataset.csv with {len(full)} rows from daily_weather_processed.csv")
    return full


if __name__ == "__main__":
    build_satellite_from_bengaluru()
    build_fertilizer_from_daily_weather()
    print("Done. Run: python satellite_model.py && python fertilizer_model.py")
