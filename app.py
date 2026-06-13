import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
import json
import os
from tensorflow.keras.models import load_model # pyright: ignore[reportMissingModuleSource]
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="AgriShield Lite", layout="centered")

# =========================
# LOAD MODELS & DATA
# =========================
model = joblib.load("risk_model.pkl")

historical_df = pd.read_csv("daily_weather_processed.csv")
historical_df["date"] = pd.to_datetime(historical_df["date"])

# Generic leaf model (includes Non_Leaf class) for gating
GENERIC_MODEL_PATH = "leaf_classifier.h5"
GENERIC_CLASSES_PATH = "leaf_classes.json"
generic_leaf_model = load_model(GENERIC_MODEL_PATH)
with open(GENERIC_CLASSES_PATH, "r") as _f:
    _generic_class_indices = json.load(_f)
generic_class_names = {int(v): k for k, v in _generic_class_indices.items()}

# Satellite / environment model (synthetic, trained on 10k samples)
SATELLITE_MODEL_PATH = "satellite_model.pkl"
satellite_bundle = None
if os.path.exists(SATELLITE_MODEL_PATH):
    satellite_bundle = joblib.load(SATELLITE_MODEL_PATH)

# Fertilizer recommendation model (synthetic, trained on 10k samples)
FERTILIZER_MODEL_PATH = "fertilizer_model.pkl"
fertilizer_bundle = None
if os.path.exists(FERTILIZER_MODEL_PATH):
    fertilizer_bundle = joblib.load(FERTILIZER_MODEL_PATH)

# Map UI crop names to specific model / class files and risk thresholds
CROP_MODEL_CONFIG = {
    "Apple": {
        "model_path": "leaf_classifier_apple.h5",
        "classes_path": "leaf_classes_apple.json",
        "risk_threshold": 0.7,
    },
    "Bell Pepper": {
        "model_path": "leaf_classifier_bell_pepper.h5",
        "classes_path": "leaf_classes_bell_pepper.json",
        "risk_threshold": 0.7,
    },
    "Cherry": {
        "model_path": "leaf_classifier_cherry.h5",
        "classes_path": "leaf_classes_cherry.json",
        "risk_threshold": 0.7,
    },
    "Corn (Maize)": {
        "model_path": "leaf_classifier_corn_maize.h5",
        "classes_path": "leaf_classes_corn_maize.json",
        "risk_threshold": 0.7,
    },
    "Grape": {
        "model_path": "leaf_classifier_grape.h5",
        "classes_path": "leaf_classes_grape.json",
        "risk_threshold": 0.7,
    },
    "Peach": {
        "model_path": "leaf_classifier_peach.h5",
        "classes_path": "leaf_classes_peach.json",
        "risk_threshold": 0.7,
    },
    "Potato": {
        "model_path": "leaf_classifier_potato.h5",
        "classes_path": "leaf_classes_potato.json",
        "risk_threshold": 0.6,
    },
    "Strawberry": {
        "model_path": "leaf_classifier_strawberry.h5",
        "classes_path": "leaf_classes_strawberry.json",
        "risk_threshold": 0.7,
    },
    "Tomato": {
        "model_path": "leaf_classifier_tomato.h5",
        "classes_path": "leaf_classes_tomato.json",
        "risk_threshold": 0.7,
    },
    # Extra crops (train with leaf_model_multi_crop.py + New Plant Diseases Dataset)
    "Blueberry": {"model_path": "leaf_classifier_blueberry.h5", "classes_path": "leaf_classes_blueberry.json", "risk_threshold": 0.7},
    "Orange": {"model_path": "leaf_classifier_orange.h5", "classes_path": "leaf_classes_orange.json", "risk_threshold": 0.7},
    "Pepper": {"model_path": "leaf_classifier_pepper.h5", "classes_path": "leaf_classes_pepper.json", "risk_threshold": 0.7},
    "Raspberry": {"model_path": "leaf_classifier_raspberry.h5", "classes_path": "leaf_classes_raspberry.json", "risk_threshold": 0.7},
    "Soybean": {"model_path": "leaf_classifier_soybean.h5", "classes_path": "leaf_classes_soybean.json", "risk_threshold": 0.7},
    "Squash": {"model_path": "leaf_classifier_squash.h5", "classes_path": "leaf_classes_squash.json", "risk_threshold": 0.7},
}

# Simple cache so models/classes are only loaded once per session
_leaf_models_cache = {}
_leaf_classnames_cache = {}


def get_leaf_model_and_classes(crop_name: str):
    """
    Load the crop-specific model and class names.
    If crop-specific files are missing (e.g. training not finished),
    fall back to the generic leaf_classifier.h5 / leaf_classes.json.
    """
    if crop_name in _leaf_models_cache and crop_name in _leaf_classnames_cache:
        return _leaf_models_cache[crop_name], _leaf_classnames_cache[crop_name]

    config = CROP_MODEL_CONFIG[crop_name]
    model_path = config["model_path"]
    classes_path = config["classes_path"]

    if os.path.exists(model_path) and os.path.exists(classes_path):
        leaf_model = load_model(model_path)
        with open(classes_path, "r") as f:
            class_indices = json.load(f)
        class_names = {int(v): k for k, v in class_indices.items()}
    else:
        # Fallback to original single model
        st.warning(
            f"Specific model for {crop_name} not found yet. "
            "Using the generic leaf model instead."
        )
        leaf_model = load_model("leaf_classifier.h5")
        with open("leaf_classes.json", "r") as f:
            class_indices = json.load(f)
        class_names = {int(v): k for k, v in class_indices.items()}

    _leaf_models_cache[crop_name] = leaf_model
    _leaf_classnames_cache[crop_name] = class_names

    return leaf_model, class_names


API_KEY = "a3eac34f9877e8fb484c36f86483636b"

# Crops and their sensitivity (for 7–10 day outlook)
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


def fetch_forecast_outlook(city_name: str, api_key: str, risk_model):
    """Fetch 5-day forecast, aggregate by day, return outlook (avg temp, total rain, avg risk, etc.)."""
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
            # Rain: 3h volume in mm; key can be "3h" or "1h" in some responses
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

# =========================
# DISEASE RECOMMENDATIONS
# =========================
disease_recommendations = {
    # Tomato
    "Tomato_Early_blight": [
        "Remove and destroy infected leaves immediately",
        "Apply Mancozeb or Chlorothalonil fungicide every 7–10 days",
        "Avoid overhead irrigation",
        "Ensure proper plant spacing",
        "Practice crop rotation",
    ],
    "Early Blight": [
        "Remove and destroy infected leaves immediately",
        "Apply Mancozeb or Chlorothalonil fungicide every 7–10 days",
        "Avoid overhead irrigation",
        "Ensure proper plant spacing",
        "Practice crop rotation",
    ],
    "Tomato_Late_blight": [
        "Isolate infected plants immediately",
        "Apply copper-based fungicide",
        "Improve field drainage",
        "Avoid working in wet fields",
        "Monitor daily during humid weather",
    ],
    "Late Blight": [
        "Isolate infected plants immediately",
        "Apply copper-based fungicide",
        "Improve field drainage",
        "Avoid working in wet fields",
        "Monitor daily during humid weather",
    ],
    "Bacterial Spot": [
        "Use copper-based bactericide",
        "Avoid overhead irrigation",
        "Remove and destroy infected plant debris",
        "Use disease-free seed and transplants",
    ],
    "Septoria Leaf Spot": [
        "Apply fungicide (chlorothalonil or mancozeb)",
        "Remove infected leaves",
        "Improve air circulation",
        "Avoid working when foliage is wet",
    ],
    "Yellow Leaf Curl Virus": [
        "Control whitefly vectors with insecticides",
        "Remove infected plants",
        "Use resistant varieties if available",
    ],
    "Tomato_healthy": [
        "Maintain regular inspection schedule",
        "Ensure balanced fertilization",
        "Avoid excessive irrigation",
    ],
    "Healthy": [
        "Maintain regular inspection schedule",
        "Ensure balanced fertilization",
        "Avoid excessive irrigation",
    ],
    # Potato
    "Potato_Early_blight": [
        "Remove infected foliage",
        "Apply Mancozeb or Azoxystrobin",
        "Ensure proper ventilation",
        "Rotate crops",
    ],
    "Potato_Late_blight": [
        "Remove infected plants immediately",
        "Apply Metalaxyl-based fungicide",
        "Improve drainage",
        "Increase monitoring during rainy weather",
    ],
    "Potato_healthy": [
        "Maintain soil moisture balance",
        "Inspect weekly",
        "Use certified seed tubers",
    ],
    # Apple
    "Apple Scab": [
        "Apply fungicide (sulfur or captan) at green tip and cover sprays",
        "Remove fallen leaves and fruit",
        "Choose resistant varieties in new plantings",
    ],
    "Black Rot": [
        "Prune out cankers and mummified fruit",
        "Apply fungicide during bloom and early cover",
        "Remove dead wood and fallen fruit",
    ],
    "Cedar Apple Rust": [
        "Apply fungicide when conditions favor infection",
        "Remove nearby juniper/cedar if feasible",
        "Use resistant apple varieties",
    ],
    # Grape (Black Rot shared with Apple; add Grape-specific if needed)
    "Esca": [
        "No curative treatment; remove severely affected vines",
        "Avoid wounding trunks; protect pruning wounds",
    ],
    "Esca (Black Measles)": [
        "No curative treatment; remove severely affected vines",
        "Avoid wounding trunks; protect pruning wounds",
    ],
    "Leaf blight (Isariopsis Leaf Spot)": [
        "Apply fungicide early in season",
        "Remove infected leaves",
        "Improve ventilation",
    ],
    # Corn
    "Cercospora Leaf Spot (Gray Leaf Spot)": [
        "Apply fungicide if needed in susceptible hybrids",
        "Rotate crops",
        "Tillage to reduce residue",
    ],
    "Common Rust": [
        "Apply fungicide in high-risk seasons",
        "Use resistant hybrids",
    ],
    "Northern Leaf Blight": [
        "Apply fungicide; use resistant hybrids",
        "Rotate and reduce residue",
    ],
    # Pepper
    "Bacterial spot": [
        "Copper bactericide; avoid overhead irrigation",
        "Use disease-free seed",
    ],
    # Cherry
    "Powdery mildew": [
        "Apply sulfur or other fungicide",
        "Improve air flow",
        "Remove severely infected leaves",
    ],
    # Peach
    "Bacterial spot": [
        "Copper sprays during dormancy and early season",
        "Choose less susceptible varieties",
    ],
    # Strawberry
    "Leaf scorch": [
        "Remove infected leaves",
        "Apply fungicide",
        "Improve drainage and air flow",
    ],
}


def _get_disease_recommendations(leaf_label: str):
    """Resolve recommendations for a predicted label (exact, normalized, or healthy fallback)."""
    if not leaf_label:
        return ["No recommendation available for this detected class."]
    # Exact match
    if leaf_label in disease_recommendations:
        return disease_recommendations[leaf_label]
    # Normalized: Tomato_Early_blight -> Early Blight style; ___ or _ to space
    normalized = leaf_label.replace("___", " ").replace("_", " ").strip()
    for key in disease_recommendations:
        if key.replace("_", " ").strip().lower() == normalized.lower():
            return disease_recommendations[key]
    # Try without crop prefix (e.g. Tomato_Early_blight -> Early blight)
    if "_" in leaf_label:
        suffix = leaf_label.split("_", 1)[-1].replace("_", " ").strip()
        for key in disease_recommendations:
            if key.lower().endswith(suffix.lower()) or suffix.lower() in key.lower():
                return disease_recommendations[key]
    # Healthy variant
    if "healthy" in leaf_label.lower():
        return disease_recommendations.get("Healthy", disease_recommendations.get("Tomato_healthy", ["Maintain regular inspection and balanced fertilization."]))
    return ["No recommendation available for this detected class."]


# =========================
# SIDEBAR: INPUTS (interactive)
# =========================
with st.sidebar:
    st.header("📍 Settings")
    crop = st.selectbox(
        "Crop",
        list(CROP_MODEL_CONFIG.keys()),
        help="Select the crop you are growing.",
    )
    city = st.text_input(
        "City / location",
        value="Bengaluru",
        help="Enter city name for weather. Use demo data if you prefer.",
    )
    demo_mode = st.checkbox(
        "Use demo weather",
        value=False,
        help="Use fixed demo values instead of live API.",
    )
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Leaf image (optional)",
        type=["jpg", "jpeg", "png"],
        help="Upload a leaf photo for disease detection and fertilizer advice.",
    )
    if uploaded_file:
        st.image(
            Image.open(uploaded_file).convert("RGB").resize((120, 120)),
            caption="Preview",
        )
    st.markdown("---")
    run_advisory = st.button("🔍 Get full advisory", type="primary", use_container_width=True)
    if "weather_probability" in st.session_state:
        st.caption(f"Last run: {st.session_state.get('last_city', '—')} • {crop}")
    if st.button("Clear results", use_container_width=True):
        for key in list(st.session_state.keys()):
            if key.startswith("advisory_") or key in ("weather_probability", "last_temp", "last_humidity", "last_rainfall", "last_pressure", "last_city", "outlook_data"):
                del st.session_state[key]
        st.rerun()

# Main area header
st.title("🌾 AgriShield Lite")
st.subheader("AI-Powered Hybrid Crop Advisory System")

# Run all predictions when button is clicked (with spinner)
if run_advisory:
    with st.spinner("Fetching weather and running models…"):
        temp = humidity = rainfall = pressure = None
        # 1) Weather & environmental risk
        if demo_mode:
            temp, humidity, rainfall, pressure = 30, 85, 8, 1008
        else:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
            try:
                data = requests.get(url).json()
                if data.get("cod") != 200:
                    st.error("Invalid city name or API issue. Check city and try again.")
                else:
                    temp = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    pressure = data["main"]["pressure"]
                    rainfall = data.get("rain", {}).get("1h", 0)
            except Exception as e:
                st.error(f"Could not fetch weather: {e}")
        if temp is not None:
            features = np.array([[temp, humidity, rainfall, pressure]])
            probability = float(model.predict_proba(features)[0][1])
            st.session_state["weather_probability"] = probability
            st.session_state["last_temp"] = float(temp)
            st.session_state["last_humidity"] = float(humidity)
            st.session_state["last_rainfall"] = float(rainfall)
            st.session_state["last_pressure"] = float(pressure)
            st.session_state["last_city"] = city
            # 2) Satellite / soil & wind (from weather)
            est_soil_type = est_soil_moisture = est_wind_speed = None
            if satellite_bundle is not None:
                X_env = np.array([[temp, humidity, rainfall, pressure]], dtype=float)
                est_soil_type = satellite_bundle["soil_model"].predict(X_env)[0]
                cond_pred = satellite_bundle["cond_model"].predict(X_env)[0]
                est_soil_moisture = float(cond_pred[0])
                est_wind_speed = float(cond_pred[1])
            st.session_state["advisory_est_soil_type"] = est_soil_type
            st.session_state["advisory_est_soil_moisture"] = est_soil_moisture
            st.session_state["advisory_est_wind_speed"] = est_wind_speed

        # 3) Leaf disease + fertilizer (if image uploaded)
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB").resize((224, 224))
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            generic_pred = generic_leaf_model.predict(img_array)
            generic_label = generic_class_names.get(int(np.argmax(generic_pred)), "")
            if generic_label == "Non_Leaf" and float(np.max(generic_pred)) >= 0.6:
                st.session_state["advisory_has_leaf"] = False
                st.session_state["advisory_non_leaf"] = True
            else:
                leaf_model, class_names = get_leaf_model_and_classes(crop)
                prediction = leaf_model.predict(img_array)
                class_index = int(np.argmax(prediction))
                confidence = float(np.max(prediction))
                leaf_label = class_names[class_index]
                weather_prob = st.session_state.get("weather_probability")
                est_soil_type = st.session_state.get("advisory_est_soil_type")
                fert_label = None
                if fertilizer_bundle is not None and weather_prob is not None and est_soil_type is not None:
                    dummies_cols = fertilizer_bundle["dummies_cols"]
                    # Crops the fertilizer model was trained on (from one-hot column names)
                    fert_crops = [c.replace("crop_", "") for c in dummies_cols if c.startswith("crop_")]
                    crop_str = str(crop).strip()
                    if crop_str not in fert_crops and fert_crops:
                        crop_str = fert_crops[0]  # fallback to first known crop
                    soil_str = str(est_soil_type).strip() if est_soil_type else "Loam"
                    # Ensure soil is one of the trained values
                    if soil_str not in ("Sandy", "Loam", "Clay"):
                        soil_str = "Loam"
                    row = {
                        "crop": crop_str,
                        "soil_type": soil_str,
                        "temperature": float(st.session_state.get("last_temp", 25)),
                        "humidity": float(st.session_state.get("last_humidity", 60)),
                        "rainfall": float(st.session_state.get("last_rainfall", 0)),
                        "risk_index": float(weather_prob),
                    }
                    X_row = pd.DataFrame([row], columns=fertilizer_bundle["feature_cols"])
                    X_enc = pd.get_dummies(X_row, columns=["crop", "soil_type"], drop_first=False)
                    # Align to exact training columns and order; fill missing with 0
                    X_enc = X_enc.reindex(columns=dummies_cols, fill_value=0)
                    fert_label = fertilizer_bundle["model"].predict(X_enc)[0]
                actions = _get_disease_recommendations(leaf_label)
                st.session_state["advisory_has_leaf"] = True
                st.session_state["advisory_non_leaf"] = False
                st.session_state["advisory_leaf_label"] = leaf_label
                st.session_state["advisory_confidence"] = confidence
                st.session_state["advisory_fert_label"] = fert_label
                st.session_state["advisory_actions"] = actions
        else:
            st.session_state["advisory_has_leaf"] = False
            st.session_state["advisory_non_leaf"] = False
    st.rerun()

# =========================
# TABS: Advisory | 7–10 day outlook | Historical | Help
# =========================
tab_advisory, tab_outlook, tab_historical, tab_help = st.tabs([
    "📋 Advisory",
    "📅 7–10 day outlook",
    "📊 Historical trends",
    "❓ How it works",
])

with tab_advisory:
    if "weather_probability" not in st.session_state:
        st.info("👆 Set **location** in the sidebar and click **Get full advisory** to see weather, risk, and field conditions.")
    else:
        prob = st.session_state["weather_probability"]
        temp = st.session_state.get("last_temp")
        humidity = st.session_state.get("last_humidity")
        rainfall = st.session_state.get("last_rainfall")
        pressure = st.session_state.get("last_pressure")
        city_used = st.session_state.get("last_city", city)

        with st.expander("🌦 Weather & environmental risk", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Temperature (°C)", f"{temp:.1f}" if temp is not None else "—")
                st.metric("Humidity (%)", f"{humidity:.0f}" if humidity is not None else "—")
            with c2:
                st.metric("Rainfall (mm)", f"{rainfall:.1f}" if rainfall is not None else "—")
                st.metric("Pressure (hPa)", f"{pressure:.0f}" if pressure is not None else "—")
            with c3:
                st.metric("Location", city_used)
                st.metric("Disease risk", f"{prob*100:.1f}%")
            st.progress(min(1.0, prob))

        est_soil = st.session_state.get("advisory_est_soil_type")
        est_sm = st.session_state.get("advisory_est_soil_moisture")
        est_wind = st.session_state.get("advisory_est_wind_speed")
        if est_soil is not None:
            with st.expander("🌍 Field conditions (estimated)", expanded=True):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.metric("Soil type", est_soil)
                with sc2:
                    st.metric("Soil moisture (0–1)", f"{est_sm:.2f}" if est_sm is not None else "—")
                with sc3:
                    st.metric("Wind speed (km/h)", f"{est_wind:.1f}" if est_wind is not None else "—")

        if st.session_state.get("advisory_non_leaf"):
            with st.expander("🌿 Leaf analysis", expanded=True):
                st.error("Uploaded image was not identified as a crop leaf. Please upload a clear leaf image.")
        elif st.session_state.get("advisory_has_leaf"):
            leaf_label = st.session_state["advisory_leaf_label"]
            confidence = st.session_state["advisory_confidence"]
            actions = st.session_state["advisory_actions"]
            fert_label = st.session_state.get("advisory_fert_label")
            risk_threshold = CROP_MODEL_CONFIG.get(crop, {}).get("risk_threshold", 0.7)

            with st.expander("🌿 Leaf disease prediction", expanded=True):
                col_leaf1, col_leaf2 = st.columns(2)
                with col_leaf1:
                    st.metric("Detected", leaf_label)
                with col_leaf2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")

            with st.expander("🌱 Verdict", expanded=True):
                if prob > risk_threshold and "healthy" in leaf_label.lower():
                    st.warning("High environmental risk. Preventive action recommended.")
                elif prob > risk_threshold and "healthy" not in leaf_label.lower():
                    st.error("High risk + disease confirmed. Immediate treatment required.")
                elif prob <= risk_threshold and "healthy" not in leaf_label.lower():
                    st.warning("Disease detected. Follow recommended actions.")
                else:
                    st.success("Crop stable. Continue monitoring.")

            if fert_label:
                with st.expander("💊 Fertilizer recommendation", expanded=True):
                    st.success(fert_label)

            with st.expander("✅ Recommended actions", expanded=True):
                for a in actions:
                    st.write("•", a)

            with st.expander("📄 Export report"):
                report_lines = [
                    "AgriShield Lite – Advisory Report",
                    "================================",
                    "",
                    f"Crop: {crop}",
                    f"Location: {city_used}",
                    f"Weather: T={temp:.1f}°C, H={humidity:.0f}%, Rain={rainfall:.1f}mm",
                    f"Environmental risk: {prob*100:.1f}%",
                    "",
                    f"Leaf: {leaf_label} (confidence: {confidence*100:.1f}%)",
                    "",
                ]
                if est_soil:
                    report_lines.append(f"Soil: {est_soil}, moisture≈{est_sm:.2f}, wind≈{est_wind:.1f} km/h")
                    report_lines.append("")
                if fert_label:
                    report_lines.append(f"Fertilizer: {fert_label}")
                    report_lines.append("")
                report_lines.append("Recommended actions:")
                for a in actions:
                    report_lines.append(f"  • {a}")
                st.download_button(
                    label="Download report (TXT)",
                    data="\n".join(report_lines),
                    file_name="agrishield_report.txt",
                    mime="text/plain",
                )
        else:
            with st.expander("🌿 Leaf & fertilizer"):
                st.info("Upload a leaf image in the sidebar and click **Get full advisory** for disease detection and fertilizer recommendation.")

with tab_outlook:
    st.markdown("**Know what to plant (or avoid) in the next 7–10 days based on your location.**")
    outlook_city = st.text_input("Location for outlook", value=st.session_state.get("last_city", city), key="outlook_city")
    get_outlook = st.button("Get 7–10 day outlook", type="primary", key="get_outlook_btn")
    if get_outlook and outlook_city:
        with st.spinner("Fetching forecast for your location…"):
            outlook = fetch_forecast_outlook(outlook_city.strip(), API_KEY, model)
        if outlook is None:
            st.error("Could not get forecast for this location. Check the city name and try again.")
        else:
            st.session_state["outlook_data"] = outlook
    if "outlook_data" in st.session_state:
        o = st.session_state["outlook_data"]
        with st.expander("📊 Next 5–7 day weather summary", expanded=True):
            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                st.metric("Location", o["city"])
                st.metric("Avg temperature", f"{o['avg_temp']:.1f} °C" if o.get("avg_temp") is not None else "—")
            with oc2:
                st.metric("Total rain (outlook)", f"{o['total_rain']:.1f} mm")
                st.metric("Avg humidity", f"{o['avg_humidity']:.0f}%" if o.get("avg_humidity") is not None else "—")
            with oc3:
                st.metric("Avg disease risk", f"{o['avg_risk']*100:.1f}%" if o.get("avg_risk") is not None else "—")
                st.metric("Peak risk (single day)", f"{o['max_risk']*100:.1f}%" if o.get("max_risk") is not None else "—")
            st.progress(min(1.0, o.get("avg_risk", 0)))

        # Build avoid list first (same thresholds for consistency), then can_plant = crops not in avoid
        avg_risk = o.get("avg_risk") or 0
        total_rain = o.get("total_rain") or 0
        avg_hum = o.get("avg_humidity") or 0
        # Thresholds: risk > 0.55 = moderate-high, rain > 25 mm over period, humidity > 78%
        risk_thresh, rain_thresh, hum_thresh = 0.55, 25.0, 78.0
        avoid = []
        for c, sens in CROP_SENSITIVITY.items():
            reasons = []
            if sens.get("risk_sensitive") and avg_risk > risk_thresh:
                reasons.append("high disease risk in the coming days")
            if sens.get("rain_sensitive") and total_rain > rain_thresh:
                reasons.append("heavy rain expected—increases rot and fungal pressure")
            if sens.get("humidity_sensitive") and avg_hum > hum_thresh:
                reasons.append("high humidity favors foliar diseases")
            if reasons:
                avoid.append((c, reasons))
        avoid_crops = {c for c, _ in avoid}
        can_plant = [c for c in CROP_SENSITIVITY if c not in avoid_crops]

        with st.expander("✅ Crops you can consider planting", expanded=True):
            if can_plant:
                st.success("Based on the next 5–7 days outlook, these crops are **suitable** to plant or maintain:")
                for c in can_plant:
                    st.write("•", c)
            else:
                st.warning("Conditions are challenging for most crops. Prefer waiting or choose only resilient crops.")
            if not can_plant and avg_risk < 0.5 and total_rain < 20:
                st.info("Apple, Cherry, Corn (Maize), Peach are often less sensitive; consider them if soil is ready.")

        with st.expander("❌ Crops to avoid (and why)", expanded=True):
            if avoid:
                st.warning("For the **next 7–10 days**, avoid or delay planting these crops:")
                for c, reasons in avoid:
                    st.write(f"• **{c}** — " + "; ".join(reasons))
            else:
                st.success("No crops are strongly discouraged for the outlook period. Still follow good practices.")

        with st.expander("📌 What you should do", expanded=True):
            actions = []
            if avg_risk > 0.55:
                actions.append("Apply preventive fungicide if you have standing crops (e.g. tomato, potato, pepper).")
                actions.append("Avoid overhead irrigation; use drip or furrow to keep foliage dry.")
            if total_rain > 25:
                actions.append("Ensure drainage is good; delay transplanting or sowing if the field is waterlogged.")
                actions.append("If you must plant, choose raised beds or well-drained spots.")
            if avg_risk <= 0.5 and total_rain < 25:
                actions.append("Good window for planting or transplanting. Prepare seedbeds and stick to your plan.")
            actions.append("Check the **Advisory** tab after 7–10 days and run **Get full advisory** again for an updated risk and leaf check.")
            if not actions:
                actions = ["Monitor weather daily. Run **Get full advisory** when you have a leaf sample or need fertilizer advice."]
            for a in actions:
                st.write("•", a)
    else:
        st.info("👆 Enter your **location** above and click **Get 7–10 day outlook** to see which crops to plant or avoid and what to do.")

with tab_historical:
    days = st.slider("Number of past days", 30, 365, 90, help="Select range for historical charts.")
    recent_data = historical_df.sort_values("date").tail(days)
    st.line_chart(
        recent_data.set_index("date")[["avg_temp", "total_rainfall", "risk"]].rename(
            columns={"avg_temp": "Temperature (°C)", "total_rainfall": "Rainfall", "risk": "Risk"}
        )
    )

with tab_help:
    st.markdown("""
    **How AgriShield Lite works**
    - **Location:** Enter a city to get live weather (or use demo data). The app estimates environmental disease risk from temperature, humidity, rainfall, and pressure.
    - **7–10 day outlook:** In the **📅 7–10 day outlook** tab, enter your location and click **Get 7–10 day outlook**. You’ll see which crops you **can** plant, which to **avoid** (and why), and **what to do** in the next 5–7 days based on forecast weather and disease risk.
    - **Field conditions:** From the same weather, a model estimates soil type, soil moisture, and wind speed for your location.
    - **Leaf image:** Upload a leaf photo to run disease detection for the selected crop. The app also recommends a fertilizer based on crop, weather, and soil.
    - **Advisory:** Open the **Advisory** tab and use **Get full advisory** in the sidebar. Expand or collapse each section to focus on what you need.
    - **Clear results:** Use **Clear results** in the sidebar to start over.
    """)
