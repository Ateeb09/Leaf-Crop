# AgriShield Lite: AI-Powered Hybrid Crop Advisory System

AgriShield Lite is an intelligent, machine learning-driven agricultural advisory application designed to help farmers and agronomists monitor crop health, assess environmental risks, predict diseases, and get tailored fertilizer recommendations.

The system utilizes a hybrid modeling approach:
1. **Environmental Disease Risk Assessment**: A machine learning model (Random Forest) that predicts the risk of crop disease outbreak based on real-time weather parameters (temperature, humidity, rainfall, and barometric pressure).
2. **Leaf Disease Classification**: Multi-crop convolutional neural network (CNN) models that classify crop leaf health and identify specific diseases from uploaded images.
3. **Satellite/Environmental Estimation**: Predicts key metrics like soil type, soil moisture, and wind speed using local weather conditions.
4. **Fertilizer Recommendation**: Suggests optimal fertilizers depending on the crop and predicted soil properties.

---

## Key Features

* **Real-time Weather Integration**: Connects to the OpenWeatherMap API to fetch current weather details for any city globally.
* **Demo Weather Mode**: Allows testing with pre-set environmental data without requiring live API calls.
* **Interactive Dashboard (Streamlit)**: A clean, user-friendly Python interface for crop advisory, image uploads, and 7-day outlook analysis.
* **Full-Stack API (Flask + React/HTML Frontend)**: A production-ready REST API backend with a high-performance frontend.
* **Deep Learning Leaf Diagnosis**: Gated classification checks (ensuring the image is actually a leaf) followed by crop-specific disease diagnosis (Tomato, Potato, Bell Pepper, Apple, Grape, Peach, Cherry, Corn, Strawberry, etc.).
* **Treatment Advisories**: Direct, actionable biological and chemical recommendations based on diagnosed diseases.

---

## Installation & Setup

Ensure you have Python 3.10+ installed.

### 1. Clone & Navigate
Navigate to the project root directory:
```bash
cd techflix
```

### 2. Set Up a Virtual Environment (Recommended)
Create and activate an isolated Python environment:

* **On Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **On macOS/Linux/Git Bash:**
  ```bash
  python -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies
Install all the required python packages:
```bash
pip install -r requirements-web.txt
pip install streamlit
```

---

## Running the Application

This project provides two distinct ways to run the interface:

### Option A: Streamlit Dashboard (Python-only, easiest for local usage)
To launch the interactive Python UI, run:
```bash
streamlit run app.py
```
* Once started, open your web browser and navigate to **`http://localhost:8501`**.
* Use the sidebar to enter a city (e.g. `Bengaluru`), select a crop, upload an optional image of a leaf, and click **"Get full advisory"**.

### Option B: Flask API Server (Production-ready API and static web page)
To run the Flask backend server:
```bash
python server.py
```
* The backend API and the static web app will be served at **`http://localhost:5000`**.

---

## Project Structure

```
├── app.py                         # Streamlit application entry point
├── server.py                      # Flask API server entry point
├── core.py                        # Common configurations, model loading, and business logic
├── requirements-web.txt           # Python library dependencies
├── README_EXTRA_CROPS.md          # Guide for downloading and training 10-15+ more crops
├── daily_weather_processed.csv    # Climate dataset used for baseline estimations
├── leaf_classifier.h5             # Generic deep learning CNN model (leaf vs non-leaf)
├── leaf_classifier_<crop>.h5      # Crop-specific trained model files (e.g., tomato, potato)
├── leaf_classes_<crop>.json       # Class label mappings for crop-specific models
├── risk_model.pkl                 # Random Forest disease risk predictor
├── satellite_model.pkl            # Soil and environment estimator model
├── fertilizer_model.pkl           # Fertilizer recommendation model
├── frontend/                      # Source code for React frontend
└── static/                        # Compiled static assets for Flask server
```

---

##  Model Training & Dataset Reorganization
If you want to train the models on additional crop datasets (such as the *New Plant Diseases Dataset* containing 38 classes):
1. Refer to the instructions inside [`README_EXTRA_CROPS.md`](README_EXTRA_CROPS.md).
2. Download the dataset and run the reorganization script:
   ```bash
   python reorganize_plant_diseases_dataset.py
   ```
3. Train models for all newly discovered crops:
   ```bash
   python leaf_model_multi_crop.py
   ```

---

## License
This project is open-source and available under the MIT License.
