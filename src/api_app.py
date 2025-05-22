import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from preprocessing import preprocess_data
from jose import jwt, JWTError

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"

app = FastAPI(title="EV Energy Forecasting API")

# Load model, scaler, and feature columns
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
model_path = os.path.join(os.path.dirname(__file__), 'cnn_gru_model.keras')
scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
feature_cols = joblib.load(os.path.join(models_dir, 'feature_cols.joblib'))
model = keras.models.load_model(model_path, compile=False)

class PredictRequest(BaseModel):
    vehicle_type: str
    station: str
    timestamp: str
    temperature: float
    charging_duration: float = 1.0

@app.post("/predict-energy")
def predict_energy(req: PredictRequest):
    # Build input DataFrame with all possible features
    data = {
        'Charging Start Time': [req.timestamp],
        'Temperature (°C)': [req.temperature],
        'Charging Duration (hours)': [req.charging_duration],
        'Vehicle Model': [req.vehicle_type],
        'Charging Station Location': [req.station],
        # Add more fields if needed
    }
    df = pd.DataFrame(data)
    # Feature engineering: cyclical encoding
    df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
    df['hour'] = df['Charging Start Time'].dt.hour
    df['dayofweek'] = df['Charging Start Time'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df.drop(['hour', 'dayofweek'], axis=1, inplace=True)
    # One-hot encode categorical features to match training
    categorical_cols = ['Vehicle Model', 'Charging Station Location']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    # Patch missing columns for model compatibility
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_cols]  # Ensure column order
    X = scaler.transform(df)
    X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
    pred = model.predict(X_reshaped).flatten()[0]
    return {"predicted_energy_kwh": float(pred)}

class ScheduleRequest(BaseModel):
    vehicle_type: str
    station: str
    date: str
    temperature: float
    urgency: int = 1  # 1=normal, 2=urgent

@app.post("/smart-schedule")
def smart_schedule(req: ScheduleRequest):
    # For demo: suggest off-peak hours if urgency is normal
    if req.urgency == 1:
        return {"best_time": "12:30 AM - 4:00 AM", "cost": 5.80, "speed": "Normal"}
    else:
        return {"best_time": "Now", "cost": 7.20, "speed": "Fast"}

class RetrainRequest(BaseModel):
    admin_token: str
    data_csv: str  # base64-encoded CSV

@app.post("/retrain-model")
def retrain_model(req: RetrainRequest):
    # JWT check (simple demo)
    try:
        payload = jwt.decode(req.admin_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Not authorized")
    except JWTError:
        raise HTTPException(status_code=403, detail="Invalid token")
    # Save and retrain logic would go here
    return {"status": "Retrain triggered (demo)"}

@app.get("/")
def root():
    return {"message": "EV Energy Forecasting API. Use /predict-energy, /smart-schedule, or /retrain-model."}
