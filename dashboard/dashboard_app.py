import streamlit as st
import requests
import pandas as pd
import datetime
import os
from tensorflow import keras
import joblib

st.set_page_config(page_title="EV Smart Charging Dashboard", layout="wide")
st.title("AI-Based EV Energy Forecasting & Smart Charging Scheduler")

TABS = ["Predict Energy Usage", "Smart Charging Suggestion", "Analytics & Admin"]
tab1, tab2, tab3 = st.tabs(TABS)

API_URL = "http://localhost:8000"

# Load the GRU model, scaler, and feature columns
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'gru_model.keras')
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.joblib'))
model = keras.models.load_model(MODEL_PATH, compile=False)

with tab1:
    st.header("Predict Energy Usage")
    with st.form("predict_form"):
        vehicle_type = st.selectbox("Vehicle Type", ["Tesla Model 3", "Nissan Leaf", "MG ZS EV", "Other"])
        station = st.text_input("Charging Station", "Delhi EV Hub")
        # Use st.date_input and st.time_input instead of st.datetime_input
        date = st.date_input("Date", datetime.date.today())
        time = st.time_input("Time", datetime.datetime.now().time())
        temperature = st.number_input("Current Temperature (°C)", value=30.0)
        charging_duration = st.slider("Charging Duration (hours)", 0.5, 8.0, 3.0, 0.5)
        submitted = st.form_submit_button("Predict")
    if submitted:
        # Combine date and time into a datetime string
        dt = datetime.datetime.combine(date, time)
        payload = {
            "vehicle_type": vehicle_type,
            "station": station,
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature": temperature,
            "charging_duration": charging_duration
        }
        resp = requests.post(f"{API_URL}/predict-energy", json=payload)
        if resp.status_code == 200:
            pred = resp.json()["predicted_energy_kwh"]
            st.success(f"Estimated energy usage for the next {charging_duration} hours: {pred:.2f} kWh")
        else:
            st.error(f"Prediction failed: {resp.text}")

with tab2:
    st.header("Smart Charging Suggestion")
    with st.form("schedule_form"):
        vehicle_type = st.selectbox("Vehicle Type", ["Tesla Model 3", "Nissan Leaf", "MG ZS EV", "Other"], key="sched_vehicle")
        station = st.text_input("Charging Station", "Delhi EV Hub", key="sched_station")
        date = st.date_input("Date", datetime.date.today(), key="sched_date")
        temperature = st.number_input("Expected Temperature (°C)", value=30.0, key="sched_temp")
        urgency = st.radio("Urgency", ["Normal", "Urgent"], index=0, key="sched_urgency")
        submitted = st.form_submit_button("Get Suggestion")
    if submitted:
        payload = {
            "vehicle_type": vehicle_type,
            "station": station,
            "date": date.strftime("%Y-%m-%d"),
            "temperature": temperature,
            "urgency": 1 if urgency == "Normal" else 2
        }
        resp = requests.post(f"{API_URL}/smart-schedule", json=payload)
        if resp.status_code == 200:
            res = resp.json()
            st.success(f"Best time to charge: {res['best_time']}\nLowest cost: ₹{res['cost']}/kWh\nCharging speed: {res['speed']}")
        else:
            st.error(f"Suggestion failed: {resp.text}")

with tab3:
    st.header("Analytics & Admin Panel")
    st.info("Upload new data, view analytics, or retrain the model (admin only). [Demo features]")
    uploaded = st.file_uploader("Upload new charging data (CSV)")
    if uploaded:
        st.write(pd.read_csv(uploaded).head())
    st.write("Model accuracy and analytics coming soon!")
    st.warning("Retrain and analytics features are placeholders in this demo.")

# When making predictions, ensure input features are processed and ordered as in API/model_train
