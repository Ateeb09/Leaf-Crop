"""Shared config and logic for AgriShield web API (no Streamlit)."""
import os
import json
import requests
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE)

CROP_MODEL_CONFIG = {
    "Apple": {"model_path": "leaf_classifier_apple.h5", "classes_path": "leaf_classes_apple.json", "risk_threshold": 0.7},
    "Bell Pepper": {"model_path": "leaf_classifier_bell_pepper.h5", "classes_path": "leaf_classes_bell_pepper.json", "risk_threshold": 0.7},
    "Cherry": {"model_path": "leaf_classifier_cherry.h5", "classes_path": "leaf_classes_cherry.json", "risk_threshold": 0.7},
    "Corn (Maize)": {"model_path": "leaf_classifier_corn_maize.h5", "classes_path": "leaf_classes_corn_maize.json", "risk_threshold": 0.7},
    "Grape": {"model_path": "leaf_classifier_grape.h5", "classes_path": "leaf_classes_grape.json", "risk_threshold": 0.7},
    "Peach": {"model_path": "leaf_classifier_peach.h5", "classes_path": "leaf_classes_peach.json", "risk_threshold": 0.7},
    "Potato": {"model_path": "leaf_classifier_potato.h5", "classes_path": "leaf_classes_potato.json", "risk_threshold": 0.6},
    "Strawberry": {"model_path": "leaf_classifier_strawberry.h5", "classes_path": "leaf_classes_strawberry.json", "risk_threshold": 0.7},
    "Tomato": {"model_path": "leaf_classifier_tomato.h5", "classes_path": "leaf_classes_tomato.json", "risk_threshold": 0.7},
    "Blueberry": {"model_path": "leaf_classifier_blueberry.h5", "classes_path": "leaf_classes_blueberry.json", "risk_threshold": 0.7},
    "Orange": {"model_path": "leaf_classifier_orange.h5", "classes_path": "leaf_classes_orange.json", "risk_threshold": 0.7},
    "Pepper": {"model_path": "leaf_classifier_pepper.h5", "classes_path": "leaf_classes_pepper.json", "risk_threshold": 0.7},
    "Raspberry": {"model_path": "leaf_classifier_raspberry.h5", "classes_path": "leaf_classes_raspberry.json", "risk_threshold": 0.7},
    "Soybean": {"model_path": "leaf_classifier_soybean.h5", "classes_path": "leaf_classes_soybean.json", "risk_threshold": 0.7},
    "Squash": {"model_path": "leaf_classifier_squash.h5", "classes_path": "leaf_classes_squash.json", "risk_threshold": 0.7},
}

API_KEY = "a3eac34f9877e8fb484c36f86483636b"

CROP_SENSITIVITY = {
    "Tomato": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Potato": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Bell Pepper": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Pepper": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Apple": {"risk_sensitive": False, "rain_sensitive": False, "humidity_sensitive": False},
    "Blueberry": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Cherry": {"risk_sensitive": False, "rain_sensitive": True, "humidity_sensitive": False},
    "Corn (Maize)": {"risk_sensitive": False, "rain_sensitive": True, "humidity_sensitive": False},
    "Grape": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Orange": {"risk_sensitive": False, "rain_sensitive": True, "humidity_sensitive": False},
    "Peach": {"risk_sensitive": False, "rain_sensitive": True, "humidity_sensitive": False},
    "Raspberry": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Soybean": {"risk_sensitive": False, "rain_sensitive": True, "humidity_sensitive": False},
    "Squash": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
    "Strawberry": {"risk_sensitive": True, "rain_sensitive": True, "humidity_sensitive": True},
}

disease_recommendations = {
    "Tomato_Early_blight": ["Remove and destroy infected leaves immediately", "Apply Mancozeb or Chlorothalonil fungicide every 7-10 days", "Avoid overhead irrigation", "Ensure proper plant spacing", "Practice crop rotation"],
    "Early Blight": ["Remove and destroy infected leaves immediately", "Apply Mancozeb or Chlorothalonil fungicide every 7-10 days", "Avoid overhead irrigation", "Ensure proper plant spacing", "Practice crop rotation"],
    "Tomato_Late_blight": ["Isolate infected plants immediately", "Apply copper-based fungicide", "Improve field drainage", "Avoid working in wet fields", "Monitor daily during humid weather"],
    "Late Blight": ["Isolate infected plants immediately", "Apply copper-based fungicide", "Improve field drainage", "Avoid working in wet fields", "Monitor daily during humid weather"],
    "Bacterial Spot": ["Use copper-based bactericide", "Avoid overhead irrigation", "Remove and destroy infected plant debris", "Use disease-free seed and transplants"],
    "Septoria Leaf Spot": ["Apply fungicide (chlorothalonil or mancozeb)", "Remove infected leaves", "Improve air circulation", "Avoid working when foliage is wet"],
    "Yellow Leaf Curl Virus": ["Control whitefly vectors with insecticides", "Remove infected plants", "Use resistant varieties if available"],
    "Tomato_healthy": ["Maintain regular inspection schedule", "Ensure balanced fertilization", "Avoid excessive irrigation"],
    "Healthy": ["Maintain regular inspection schedule", "Ensure balanced fertilization", "Avoid excessive irrigation"],
    "Potato_Early_blight": ["Remove infected foliage", "Apply Mancozeb or Azoxystrobin", "Ensure proper ventilation", "Rotate crops"],
    "Potato_Late_blight": ["Remove infected plants immediately", "Apply Metalaxyl-based fungicide", "Improve drainage", "Increase monitoring during rainy weather"],
    "Potato_healthy": ["Maintain soil moisture balance", "Inspect weekly", "Use certified seed tubers"],
    "Apple Scab": ["Apply fungicide (sulfur or captan) at green tip and cover sprays", "Remove fallen leaves and fruit", "Choose resistant varieties in new plantings"],
    "Black Rot": ["Prune out cankers and mummified fruit", "Apply fungicide during bloom and early cover", "Remove dead wood and fallen fruit"],
    "Cedar Apple Rust": ["Apply fungicide when conditions favor infection", "Remove nearby juniper/cedar if feasible", "Use resistant apple varieties"],
    "Esca": ["No curative treatment; remove severely affected vines", "Avoid wounding trunks; protect pruning wounds"],
    "Esca (Black Measles)": ["No curative treatment; remove severely affected vines", "Avoid wounding trunks; protect pruning wounds"],
    "Leaf blight (Isariopsis Leaf Spot)": ["Apply fungicide early in season", "Remove infected leaves", "Improve ventilation"],
    "Cercospora Leaf Spot (Gray Leaf Spot)": ["Apply fungicide if needed in susceptible hybrids", "Rotate crops", "Tillage to reduce residue"],
    "Common Rust": ["Apply fungicide in high-risk seasons", "Use resistant hybrids"],
    "Northern Leaf Blight": ["Apply fungicide; use resistant hybrids", "Rotate and reduce residue"],
    "Bacterial spot": ["Copper bactericide; avoid overhead irrigation", "Use disease-free seed"],
    "Powdery mildew": ["Apply sulfur or other fungicide", "Improve air flow", "Remove severely infected leaves"],
    "Leaf scorch": ["Remove infected leaves", "Apply fungicide", "Improve drainage and air flow"],
}

_leaf_models_cache = {}
_leaf_classnames_cache = {}


def get_leaf_model_and_classes(crop_name: str):
    config = CROP_MODEL_CONFIG.get(crop_name, CROP_MODEL_CONFIG.get("Tomato"))
    if crop_name in _leaf_models_cache and crop_name in _leaf_classnames_cache:
        return _leaf_models_cache[crop_name], _leaf_classnames_cache[crop_name]
    model_path = config["model_path"]
    classes_path = config["classes_path"]
    if os.path.exists(model_path) and os.path.exists(classes_path):
        leaf_model = load_model(model_path)
        with open(classes_path) as f:
            class_indices = json.load(f)
        class_names = {int(v): k for k, v in class_indices.items()}
    else:
        leaf_model = load_model("leaf_classifier.h5")
        with open("leaf_classes.json") as f:
            class_indices = json.load(f)
        class_names = {int(v): k for k, v in class_indices.items()}
    _leaf_models_cache[crop_name] = leaf_model
    _leaf_classnames_cache[crop_name] = class_names
    return leaf_model, class_names


def _get_disease_recommendations(leaf_label: str):
    if not leaf_label:
        return ["No recommendation available for this detected class."]
    if leaf_label in disease_recommendations:
        return disease_recommendations[leaf_label]
    normalized = leaf_label.replace("___", " ").replace("_", " ").strip()
    for key in disease_recommendations:
        if key.replace("_", " ").strip().lower() == normalized.lower():
            return disease_recommendations[key]
    if "_" in leaf_label:
        suffix = leaf_label.split("_", 1)[-1].replace("_", " ").strip()
        for key in disease_recommendations:
            if key.lower().endswith(suffix.lower()) or suffix.lower() in key.lower():
                return disease_recommendations[key]
    if "healthy" in leaf_label.lower():
        return disease_recommendations.get("Healthy", ["Maintain regular inspection and balanced fertilization."])
    return ["No recommendation available for this detected class."]


def fetch_forecast_outlook(city_name: str, api_key: str, risk_model):
    url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={api_key}&units=metric"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("cod") not in (200, "200"):
            return None
        list_items = data.get("list") or []
        daily = {}
        for item in list_items:
            dt_txt = item.get("dt_txt") or ""
            dt = dt_txt[:10] if len(dt_txt) >= 10 else ""
            if not dt:
                continue
            main = item.get("main") or {}
            if dt not in daily:
                daily[dt] = {"temp": [], "humidity": [], "pressure": [], "rain": []}
            daily[dt]["temp"].append(main.get("temp", 0))
            daily[dt]["humidity"].append(main.get("humidity", 0))
            daily[dt]["pressure"].append(main.get("pressure", 1013))
            rain_obj = item.get("rain") or {}
            rain_mm = rain_obj.get("3h", rain_obj.get("1h", 0))
            daily[dt]["rain"].append(float(rain_mm) if rain_mm is not None else 0)
        sorted_dates = sorted(daily.keys())[:7]
        if not sorted_dates:
            return None
        temps, hums, rains, risks = [], [], [], []
        for dt in sorted_dates:
            d = daily[dt]
            n = len(d["temp"])
            if n == 0:
                continue
            t = sum(d["temp"]) / n
            h = sum(d["humidity"]) / n
            p = sum(d["pressure"]) / n
            r = sum(d["rain"])
            temps.append(t)
            hums.append(h)
            rains.append(r)
            feat = np.array([[float(t), float(h), float(r), float(p)]])
            risks.append(float(risk_model.predict_proba(feat)[0][1]))
        return {
            "city": city_name,
            "days": len(sorted_dates),
            "avg_temp": sum(temps) / len(temps) if temps else None,
            "total_rain": sum(rains),
            "avg_humidity": sum(hums) / len(hums) if hums else None,
            "avg_risk": sum(risks) / len(risks) if risks else None,
            "max_risk": max(risks) if risks else None,
            "sorted_dates": sorted_dates,
        }
    except Exception:
        return None
