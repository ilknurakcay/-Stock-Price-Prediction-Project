# NVIDIA Stock Price Prediction Using LSTM

## Project Overview
This project implements a deep learning model using LSTM (Long Short-Term Memory) networks to predict NVIDIA (NVDA) stock prices. The model incorporates various technical indicators and uses hyperparameter optimization to achieve better prediction accuracy.

## Features
- Implements multiple technical indicators:
  - Relative Strength Index (RSI)xxxx
  - Moving Average Convergence Divergence (MACD)
  - Bollinger Bands
  - Money Flow Index (MFI)
  - On-Balance Volume (OBV)
  - Fibonacci Retracement Levels
  - Stochastic Oscillator
- Hyperparameter optimization using Optuna
- Time series data preprocessing
- Model evaluation with comprehensive metrics

## Technical Indicators
The project calculates and uses the following technical indicators for prediction:
- **RSI**: 14-day period calculation of relative strength
- **MACD**: Using 12-day and 26-day EMAs with 9-day signal line
- **Bollinger Bands**: 20-day period with 2 standard deviations
- **MFI**: 14-day period money flow calculations
- **OBV**: Volume-price relationship indicator
- **Fibonacci Levels**: 23.6%, 38.2%, 50.0%, and 61.8% retracement levels
- **Stochastic Oscillator**: 14-day period with 3-day SMA smoothing

## Project Structure
```
├── best_params.yaml         # Best hyperparameters from optimization
├── hyperparameter.py       # Hyperparameter optimization using Optuna
├── predict_34.py           # Main prediction script
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
python predict_34.py
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
- Directional Accuracy
- Correlation
- Maximum Drawdown

## Contributing
Feel free to fork the project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[Choose an appropriate license for your project]

## Author
[Your Name]

## Acknowledgments
- Data source: Kaggle S&P 500 dataset
- Technical indicators implementation based on standard financial formulas
