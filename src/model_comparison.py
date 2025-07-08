import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessing import load_data, preprocess_data
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(42)
tf.random.set_seed(42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def build_ann(input_shape):
    inputs = keras.Input(shape=(input_shape, 1))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_rnn(input_shape):
    inputs = keras.Input(shape=(input_shape, 1))
    x = keras.layers.SimpleRNN(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.SimpleRNN(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_lstm(input_shape):
    inputs = keras.Input(shape=(input_shape, 1))
    x = keras.layers.LSTM(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-3), dropout=0.3)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LSTM(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-3), dropout=0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

def build_gru(input_shape):
    inputs = keras.Input(shape=(input_shape, 1))
    x = keras.layers.GRU(64, return_sequences=True, kernel_regularizer=keras.regularizers.l2(1e-3), dropout=0.3)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(32, return_sequences=False, kernel_regularizer=keras.regularizers.l2(1e-3), dropout=0.3)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mse', metrics=['mae'])
    return model

def train_and_evaluate(model_fn, X_train, y_train, X_test, y_test, name):
    model = model_fn(X_train.shape[1])
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0, callbacks=[early_stop, reduce_lr])
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()
    results = {
        'model': name,
        'train_rmse': rmse(y_train, y_pred_train),
        'train_mape': mape(y_train, y_pred_train),
        'test_rmse': rmse(y_test, y_pred_test),
        'test_mape': mape(y_test, y_pred_test)
    }
    return results, history

def add_advanced_features(df):
    import pandas as pd
    # Temporal features
    if 'Charging Start Time' in df.columns:
        dt = pd.to_datetime(df['Charging Start Time'])
        df['month'] = dt.dt.month
        df['dayofweek'] = dt.dt.dayofweek
        df['hour'] = dt.dt.hour
        df['is_weekend'] = dt.dt.dayofweek >= 5
    # Lag features (per user)
    if 'User ID' in df.columns and 'Charging Start Time' in df.columns:
        df = df.sort_values(['User ID', 'Charging Start Time'])
        df['prev_energy'] = df.groupby('User ID')['Energy Consumed (kWh)'].shift(1)
        df['time_since_last_charge'] = (
            pd.to_datetime(df['Charging Start Time']) -
            pd.to_datetime(df.groupby('User ID')['Charging Start Time'].shift(1))
        ).dt.total_seconds() / 3600
        df = df.sort_index()
    # Frequency encoding
    for col in ['User ID', 'Vehicle Model', 'Location']:
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq)
    # Interaction features
    if 'Charging Rate (kW)' in df.columns and 'Charging Duration (hours)' in df.columns:
        df['rate_x_duration'] = df['Charging Rate (kW)'] * df['Charging Duration (hours)']
    # Fill NaNs in engineered features
    engineered_cols = ['month','dayofweek','hour','is_weekend','prev_energy','time_since_last_charge','User ID_freq','Vehicle Model_freq','Location_freq','rate_x_duration']
    for col in engineered_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df

def baseline_mean_predictor(y_train, y_test):
    mean_pred = np.mean(y_train)
    y_pred = np.full_like(y_test, mean_pred)
    return {
        'model': 'MeanBaseline',
        'train_rmse': 0.0,
        'train_mape': 0.0,
        'test_rmse': rmse(y_test, y_pred),
        'test_mape': mape(y_test, y_pred)
    }

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ev_charging_patterns.csv')
    df = load_data(data_path)
    df = add_advanced_features(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X, y, scaler, feature_cols = preprocess_data(df, target_col='Energy Consumed (kWh)', time_col='Charging Start Time')
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    models = [
        (build_ann, 'ANN'),
        (build_rnn, 'RNN'),
        (build_lstm, 'LSTM'),
        (build_gru, 'GRU')
    ]
    results = []
    # Baseline
    baseline = baseline_mean_predictor(y_train, y_test)
    print(f"Baseline (mean) - Test RMSE: {baseline['test_rmse']:.2f}, Test MAPE: {baseline['test_mape']:.2f}%")
    results.append(baseline)
    for fn, name in models:
        print(f'Training {name}...')
        res, _ = train_and_evaluate(fn, X_train, y_train, X_test, y_test, name)
        print(f"{name} - Train RMSE: {res['train_rmse']:.2f}, Train MAPE: {res['train_mape']:.2f}%, Test RMSE: {res['test_rmse']:.2f}, Test MAPE: {res['test_mape']:.2f}%")
        results.append(res)
    print('\nModel Comparison Table:')
    print('| Model | Train RMSE | Train MAPE | Test RMSE | Test MAPE |')
    print('|-------|------------|------------|-----------|-----------|')
    for r in results:
        print(f"| {r['model']} | {r['train_rmse']:.2f} | {r['train_mape']:.2f}% | {r['test_rmse']:.2f} | {r['test_mape']:.2f}% |")

if __name__ == "__main__":
    main()
