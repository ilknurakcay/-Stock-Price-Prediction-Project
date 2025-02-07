# NVIDIA Stock Price Prediction Using LSTM

## Project Overview
This project implements a deep learning model using LSTM (Long Short-Term Memory) networks to predict NVIDIA (NVDA) stock prices. The model incorporates various technical indicators and uses hyperparameter optimization to achieve better prediction accuracy.

## Overview
- Implements multiple technical indicators:
  - Relative Strength Index (RSI)
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Money Flow Index (MFI)
  - On-Balance Volume (OBV)
  - Fibonacci Retracement Levels
  - Stochastic Oscillator
- Time series data preprocessing
- Hyperparameter optimization using Optuna
- Model evaluation with comprehensive metrics


## Project Structure
```
├── best_params.yaml         # Best hyperparameters from optimization
├── hyperparameter.py       # Hyperparameter optimization using Optuna
├── predict.py              # Main prediction script
└── preprocess.py           # Data preprocessing and feature engineering
```

## Requirements
- Python 3.x
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Optuna
- Matplotlib
- PyYAML

## Installation
1. Clone the repository:
```bash
git clone [your-repository-url]
```

2. Install required packages:
```bash
pip install tensorflow pandas numpy scikit-learn optuna matplotlib pyyaml
```

## Usage
1. First, run the preprocessing script:
```bash
python preprocess.py
```

2. Optimize hyperparameters (optional):
```bash
python hyperparameter.py
```

3. Run the prediction model:
```bash
python predict.py
```

## Model Architecture
- LSTM layers with configurable units and layers
- Dropout layers for regularization
- Dense output layer
- Adam optimizer with configurable learning rate

## Model Evaluation Metrics
The model's performance is evaluated using:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²)
- Maximum Drawdown



## Acknowledgments
- Data source: Kaggle S&P 500 dataset
- Technical indicators implementation based on standard financial formulas
