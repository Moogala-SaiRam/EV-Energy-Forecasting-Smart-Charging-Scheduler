import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model

# --- Load trained model and scalers ---
model = load_model("ev_crnn_gru_model.h5", compile=False)  # <-- add compile=False
model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # <-- recompile
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# --- Streamlit UI ---
st.title("🔋 EV Charging kWh Forecast App")
st.markdown("Enter the date, start time, and duration to forecast energy consumption.")

# Input fields
input_date = st.date_input("Select Date")
start_time = st.time_input("Select Start Time")
charge_duration = st.number_input("Charging Duration (in hours)", min_value=0.25, max_value=24.0, step=0.25)

if st.button("Predict Energy Consumption"):
    try:
        # Convert inputs to datetime
        created_dt = datetime.combine(input_date, start_time)
        ended_dt = created_dt + timedelta(hours=charge_duration)

        # Convert to UNIX timestamps
        created_ts = int(created_dt.timestamp())
        ended_ts = int(ended_dt.timestamp())
        weekday = created_dt.weekday()

        # Construct input vector
        input_data = pd.DataFrame([{
            'sessionId': 999999,        # Dummy sessionId
            'dollars': 4.5,             # Average price (can be refined)
            'created_ts': created_ts,
            'ended_ts': ended_ts,
            'chargeTimeHrs': charge_duration,
            'weekday': weekday,
            'userId': 10,
            'stationId': 101,
            'locationId': 5
        }])

        # Scale and reshape input
        X_scaled = scaler_X.transform(input_data)
        X_seq = np.expand_dims(X_scaled, axis=1)

        # Predict
        y_pred_scaled = model.predict(X_seq)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predicted_kwh = round(y_pred[0][0], 2)

        st.success(f"⚡ Predicted Energy Consumption: **{predicted_kwh} kWh**")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
