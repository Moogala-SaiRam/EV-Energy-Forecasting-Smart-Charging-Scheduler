import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(df, target_col, time_col, scale_target=False):
    # Fill missing values
    df = df.ffill().bfill()

    # Time-based feature engineering
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['dayofweek'] = df[time_col].dt.dayofweek

    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df.drop(['hour', 'dayofweek'], axis=1, inplace=True)

    # One-hot encode categorical features
    categorical_cols = ['Vehicle Model', 'Charging Station Location', 'Charger Type', 'User Type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    # Define features (exclude non-numeric and irrelevant)
    exclude_cols = [target_col, time_col, 'User ID', 'Charging Station ID', 'Charging End Time']
    feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])

    # Prepare target
    y = df[target_col].values
    target_scaler = None
    if scale_target:
        target_scaler = StandardScaler()
        y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Reshape for CNN/GRU: (samples, timesteps, features=1)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Create save directory
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(save_dir, exist_ok=True)

    # Save scalers and feature config
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(feature_cols, os.path.join(save_dir, 'feature_cols.joblib'))
    if scale_target:
        joblib.dump(target_scaler, os.path.join(save_dir, 'target_scaler.joblib'))

    print(f"[INFO] Preprocessing complete. Feature shape: {X.shape}, Target shape: {y.shape}")
    return X, y, scaler, feature_cols

def load_data(path):
    return pd.read_csv(path)
