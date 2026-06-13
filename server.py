"""
AgriShield web app: Flask API + HTML/CSS/JS frontend.
Run from project root: python server.py
Open: http://localhost:5000
"""
import os
import io
import json
import joblib
import numpy as np
import pandas as pd
import requests
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory

import core

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

app = Flask(__name__, static_folder="static", static_url_path="")

print("Loading models...")
risk_model = joblib.load("risk_model.pkl")
historical_df = pd.read_csv("daily_weather_processed.csv")
historical_df["date"] = pd.to_datetime(historical_df["date"])

from tensorflow.keras.models import load_model as keras_load_model
generic_leaf_model = keras_load_model("leaf_classifier.h5")
with open("leaf_classes.json") as f:
    generic_class_names = {int(v): k for k, v in json.load(f).items()}

satellite_bundle = joblib.load("satellite_model.pkl") if os.path.exists("satellite_model.pkl") else None
fertilizer_bundle = joblib.load("fertilizer_model.pkl") if os.path.exists("fertilizer_model.pkl") else None
print("Models loaded.")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/crops")
def api_crops():
    return jsonify(list(core.CROP_MODEL_CONFIG.keys()))


@app.route("/api/advisory", methods=["POST"])
def api_advisory():
    city = (request.form.get("city") or (request.is_json and request.json.get("city"))) or "Bengaluru"
    demo = str(request.form.get("demo", "") or (request.is_json and request.json.get("demo") or "")).lower() in ("1", "true", "yes")
    crop = (request.form.get("crop") or (request.is_json and request.json.get("crop"))) or "Tomato"
    file = request.files.get("image")

    temp = humidity = rainfall = pressure = None
    if demo:
        temp, humidity, rainfall, pressure = 30.0, 85.0, 8.0, 1008.0
    else:
        try:
            r = requests.get(
                f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={core.API_KEY}&units=metric",
                timeout=10,
            )
            data = r.json()
            if data.get("cod") not in (200, "200"):
                return jsonify({"error": "Invalid city or API issue"}), 400
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            rainfall = data.get("rain", {}).get("1h", 0) or 0
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    features = np.array([[temp, humidity, rainfall, pressure]], dtype=float)
    prob = float(risk_model.predict_proba(features)[0][1])

    est_soil_type = est_soil_moisture = est_wind_speed = None
    if satellite_bundle is not None:
        X_env = np.array([[temp, humidity, rainfall, pressure]], dtype=float)
        est_soil_type = satellite_bundle["soil_model"].predict(X_env)[0]
        cond_pred = satellite_bundle["cond_model"].predict(X_env)[0]
        est_soil_moisture = float(cond_pred[0])
        est_wind_speed = float(cond_pred[1])

    out = {
        "city": city,
        "crop": crop,
        "weather": {"temp": temp, "humidity": humidity, "rainfall": rainfall, "pressure": pressure},
        "risk": prob,
        "field": {"soil_type": est_soil_type, "soil_moisture": est_soil_moisture, "wind_speed": est_wind_speed},
        "leaf": None,
        "verdict": None,
        "fertilizer": None,
        "actions": [],
    }

    if file and file.filename:
        try:
            img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            pred = generic_leaf_model.predict(arr)
            generic_label = generic_class_names.get(int(np.argmax(pred)), "")
            if generic_label == "Non_Leaf" and float(np.max(pred)) >= 0.6:
                out["leaf"] = {"non_leaf": True}
            else:
                leaf_model, class_names = core.get_leaf_model_and_classes(crop)
                pred2 = leaf_model.predict(arr)
                idx = int(np.argmax(pred2))
                conf = float(np.max(pred2))
                leaf_label = class_names[idx]
                actions = core._get_disease_recommendations(leaf_label)
                fert_label = None
                if fertilizer_bundle and est_soil_type is not None:
                    dummies_cols = fertilizer_bundle["dummies_cols"]
                    fert_crops = [c.replace("crop_", "") for c in dummies_cols if c.startswith("crop_")]
                    crop_str = str(crop).strip()
                    if crop_str not in fert_crops and fert_crops:
                        crop_str = fert_crops[0]
                    soil_str = str(est_soil_type).strip() if est_soil_type else "Loam"
                    if soil_str not in ("Sandy", "Loam", "Clay"):
                        soil_str = "Loam"
                    row = {
                        "crop": crop_str,
                        "soil_type": soil_str,
                        "temperature": float(temp),
                        "humidity": float(humidity),
                        "rainfall": float(rainfall),
                        "risk_index": float(prob),
                    }
                    X_row = pd.DataFrame([row], columns=fertilizer_bundle["feature_cols"])
                    X_enc = pd.get_dummies(X_row, columns=["crop", "soil_type"], drop_first=False)
                    X_enc = X_enc.reindex(columns=dummies_cols, fill_value=0)
                    fert_label = fertilizer_bundle["model"].predict(X_enc)[0]
                thr = core.CROP_MODEL_CONFIG.get(crop, {}).get("risk_threshold", 0.7)
                if prob > thr and "healthy" in leaf_label.lower():
                    verdict, verdict_text = "warning", "High environmental risk. Preventive action recommended."
                elif prob > thr and "healthy" not in leaf_label.lower():
                    verdict, verdict_text = "danger", "High risk + disease confirmed. Immediate treatment required."
                elif prob <= thr and "healthy" not in leaf_label.lower():
                    verdict, verdict_text = "warning", "Disease detected. Follow recommended actions."
                else:
                    verdict, verdict_text = "success", "Crop stable. Continue monitoring."
                out["leaf"] = {"label": leaf_label, "confidence": conf, "non_leaf": False}
                out["verdict"] = {"level": verdict, "text": verdict_text}
                out["fertilizer"] = fert_label
                out["actions"] = actions
        except Exception as e:
            out["leaf"] = {"error": str(e)}

    return jsonify(out)


@app.route("/api/outlook", methods=["POST"])
def api_outlook():
    data = request.json or request.form or {}
    city = (data.get("city") or request.form.get("city") or "Bengaluru").strip()
    outlook = core.fetch_forecast_outlook(city, core.API_KEY, risk_model)
    if outlook is None:
        return jsonify({"error": "Could not get forecast for this location"}), 400
    avg_risk = outlook.get("avg_risk") or 0
    total_rain = outlook.get("total_rain") or 0
    avg_hum = outlook.get("avg_humidity") or 0
    risk_thresh, rain_thresh, hum_thresh = 0.55, 25.0, 78.0
    avoid = []
    for c, sens in core.CROP_SENSITIVITY.items():
        reasons = []
        if sens.get("risk_sensitive") and avg_risk > risk_thresh:
            reasons.append("high disease risk in the coming days")
        if sens.get("rain_sensitive") and total_rain > rain_thresh:
            reasons.append("heavy rain expected")
        if sens.get("humidity_sensitive") and avg_hum > hum_thresh:
            reasons.append("high humidity favors foliar diseases")
        if reasons:
            avoid.append({"crop": c, "reasons": reasons})
    avoid_crops = {a["crop"] for a in avoid}
    can_plant = [c for c in core.CROP_SENSITIVITY if c not in avoid_crops]
    actions = []
    if avg_risk > 0.55:
        actions.append("Apply preventive fungicide if you have standing crops.")
        actions.append("Avoid overhead irrigation; use drip or furrow.")
    if total_rain > 25:
        actions.append("Ensure drainage is good; delay transplanting if waterlogged.")
    if avg_risk <= 0.5 and total_rain < 25:
        actions.append("Good window for planting or transplanting.")
    actions.append("Check the Advisory after 7-10 days for an updated risk.")
    return jsonify({"outlook": outlook, "can_plant": can_plant, "avoid": avoid, "actions": actions})


@app.route("/api/historical")
def api_historical():
    days = min(365, max(30, int(request.args.get("days", 90))))
    df = historical_df.sort_values("date").tail(days)
    return jsonify({
        "dates": df["date"].astype(str).tolist(),
        "temperature": df["avg_temp"].tolist(),
        "rainfall": df["total_rainfall"].tolist(),
        "risk": df["risk"].tolist(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
