import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocessing import load_data, preprocess_data
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress most TF warnings
np.random.seed(42)
tf.random.set_seed(42)

def build_gru_best(input_shape):
    inputs = keras.Input(shape=(input_shape, 1))
    x = keras.layers.GRU(128, return_sequences=True, dropout=0.3, kernel_regularizer=keras.regularizers.l2(1e-3))(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(64, return_sequences=False, dropout=0.3, kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-3))(x)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

def main():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'ev_charging_patterns.csv')
    df = load_data(data_path)
    # Shuffle data for better generalization
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X, y, scaler, feature_cols = preprocess_data(df, target_col='Energy Consumed (kWh)', time_col='Charging Start Time')
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = build_gru_best(X.shape[1])
    print('Training GRU model (best practices)...')
    # Professional standard: batch_size=32, epochs=100, callbacks for best fit
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stop, reduce_lr])
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test MSE: {loss:.2f}, MAE: {mae:.2f}")
    # Show training/validation loss and MAE
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_history.png'))
    plt.close()
    model.save(os.path.join(os.path.dirname(__file__), 'gru_model.keras'))
    print('Model saved as gru_model.keras')
    print('Training history plot saved as training_history.png')

if __name__ == "__main__":
    main()
