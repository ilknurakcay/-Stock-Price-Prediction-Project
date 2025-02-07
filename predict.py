import pandas as pd
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import yaml
import os
import warnings


# Default parameters 
default_params = {
    'units': 64,
    'dropout_rate': 0.01,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 38,
    'lstm_layers': 2,
    'activation': 'relu',
    'window_size': 7
}

# Try to load best params from YAML, fallback to defaults
try:
    with open('best_params.yaml', 'r') as f:
        params = yaml.safe_load(f)
        print("Loaded parameters from best_params.yaml")
        
    # Validate loaded parameters
    valid_params = {}
    for key in default_params:
        if key in params:
            valid_params[key] = params[key]
        else:
            warnings.warn(f"Missing {key} in YAML, using default: {default_params[key]}")
            valid_params[key] = default_params[key]
            
except FileNotFoundError:
    warnings.warn("best_params.yaml not found, using default parameters")
    valid_params = default_params.copy()
except Exception as e:
    warnings.warn(f"Error loading YAML: {str(e)}, using default parameters")
    valid_params = default_params.copy()

# Extract parameters (safe for missing keys)
units = valid_params.get('units', default_params['units'])
drop_rate = valid_params.get('dropout_rate', default_params['dropout_rate'])
learning_rate = valid_params.get('learning_rate', default_params['learning_rate'])
batch_size = valid_params.get('batch_size', default_params['batch_size'])
epochs = valid_params.get('epochs', default_params['epochs'])
lstm_layers = valid_params.get('lstm_layers', default_params['lstm_layers'])
activation = valid_params.get('activation', default_params['activation'])
window_size = valid_params.get('window_size', default_params['window_size'])


def create_model(units, dropout_rate, learning_rate, batch_size, epochs, lstm_layers, activation):
    """
    Creates LSTM model with configurable layers, units, and hyperparameters
    """
    model = Sequential()
    for i in range(lstm_layers):
        if i == 0:
            model.add(LSTM(units=units, activation=activation, return_sequences=True, input_shape=(window_size, X_train_scaled.shape[1])))
        elif i == lstm_layers - 1:
            model.add(LSTM(units=units, activation=activation))  
        else:
            model.add(LSTM(units=units, activation=activation, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1))

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    return model


X_train_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_train_scaled")
X_test_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_test_scaled")
y_train_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_train_scaled")
y_test_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_test_scaled")
y_test =  pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_test")
y_train =  pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_train")
window_size = 7  
generator = TimeseriesGenerator(np.array(X_train_scaled), np.array(y_train_scaled), length=window_size, batch_size=64)
test_generator = TimeseriesGenerator(np.array(X_test_scaled), np.array(y_test_scaled), length=window_size, batch_size=32)
final_model = create_model(units=units,
                          dropout_rate=drop_rate,
                          learning_rate=learning_rate,
                          batch_size=batch_size,
                          epochs=epochs,
                          lstm_layers=lstm_layers,
                          activation=activation)


final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
# Adds early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
history = final_model.fit(generator, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[early_stopping])
predictions = final_model.predict(test_generator)

# Creates scaler for target variable
target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

#convert scaled predictions back to their original values.
y_test_original = target_scaler.inverse_transform(y_test_scaled[window_size:])  
predicted_values = target_scaler.inverse_transform(predictions)

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive metrics 
        y_true (np.array): Actual values
        y_pred (np.array): Predicted values
    """
    metrics = {}
    
    metrics['MAE'] = mean_absolute_error(y_true, y_pred)
    metrics['MSE'] = mean_squared_error(y_true, y_pred)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAPE'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    metrics['R2'] = r2_score(y_true, y_pred)
    
    
    cumulative_returns = (y_pred / y_true - 1)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / (1 + peak)
    metrics['Max_Drawdown'] = np.min(drawdown) * 100
    
    return metrics

results = calculate_metrics(y_test_original, predicted_values)

for metric, value in results.items():
    print(f"{metric}: {value}")


def plot_resuts(y_test_original, predicted_values):

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_original, label="Real Close Price", color='blue')
    plt.plot(predicted_values, label="Predicted Close Price", color='red', linestyle="dashed")
    plt.xlabel("Time")
    plt.ylabel("Close Price")
    plt.title("Real vs Predicted Close Price")
    plt.legend()
    plt.show()

def plot_loss(history):
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_loss(history)
plot_resuts(y_test_original, predicted_values)