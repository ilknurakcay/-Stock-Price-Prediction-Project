import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Only show error messages
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import yaml
import optuna


X_train = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_train")
X_test = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_test")
y_train = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_train")
y_test = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_test")


#scales values between 0 and 1 for X.
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#scales values between 0 and 1 for y.
target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

X_train_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_train_scaled")
X_test_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/X_test_scaled")
y_train_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_train_scaled")
y_test_scaled = pd.read_csv("/Users/ilknurakcay/Desktop/finance/y_test_scaled")


# Define the window size (number of past days used for prediction)
window_size = 7  
# Create a time series data generator for training data
generator = TimeseriesGenerator(np.array(X_train_scaled), np.array(y_train_scaled), length=window_size, batch_size=64)
# Create a time series data generator for test data
test_generator = TimeseriesGenerator(np.array(X_test_scaled), np.array(y_test_scaled), length=window_size, batch_size=32)

def create_model(units, dropout_rate, learning_rate, batch_size, epochs, lstm_layers, activation):
    """
    Creates LSTM model with  layers, units, and other hyperparameters
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



# Hyperparameter optimization with Optuna
def objective(trial):
    units = trial.suggest_int('units', 128, 256, 512)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.3)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)  
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64,128])
    epochs = trial.suggest_categorical('epochs', [50]) #Source problem 
    lstm_layers = trial.suggest_int('lstm_layers', 1, 5)  
    activation = trial.suggest_categorical('activation', ['tanh', 'relu'])  

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
    # Adds early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)  

    history = model.fit(generator, epochs=epochs, batch_size=batch_size, validation_data=test_generator, callbacks=[early_stopping], verbose=0)  

    # Test seti üzerinde değerlendirme
    loss = model.evaluate(test_generator, verbose=0)
    return loss

study = optuna.create_study(direction='minimize')  
study.optimize(objective, n_trials=100)  #n trails

print("The Best Paramaters: ", study.best_params)
print("The Best Scores (MSE): ", study.best_value)

#Prediction with best model
best_model = create_model(**study.best_params) 
best_model.compile(optimizer=keras.optimizers.Adam(learning_rate=study.best_params['learning_rate']), loss='mse') 
best_model.fit(generator, epochs=study.best_params['epochs'], batch_size=study.best_params['batch_size'], verbose=0)  

# Prediction on test set
predictions = best_model.predict(test_generator)

test_loss = best_model.evaluate(test_generator)
print(f"Test Loss (MSE): {test_loss}")


with open('best_params.yaml', 'w') as f:
    yaml.safe_dump(study.best_params, f, sort_keys=False)

print("Saved best parameters to best_params.yaml")

