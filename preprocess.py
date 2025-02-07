import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

df = pd.read_csv('/Users/ilknurakcay/.cache/kagglehub/datasets/camnugent/sandp500/versions/4/all_stocks_5yr.csv')


unique_name = df["Name"].unique()
NVDA_data = df[df['Name'] == 'NVDA']

# RSI calculation formula
def calculate_rsi(data, window=14):
    """
    It is usually calculated over a 14-day period 
    """

    #Calculation diffeerence of close price
    delta = data['close'].diff()
    
    gain = delta.where(delta > 0, 0)  
    loss = -delta.where(delta < 0, 0)  #
    avg_gain = gain.rolling(window=window, min_periods=1).mean() 
    avg_loss = loss.rolling(window=window, min_periods=1).mean() 

    # Calculation of RS (Relative Strength) 
    rs = avg_gain / avg_loss

    # Calculation of RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Dataset selected as NVDA
NVDA_data['RSI'] = calculate_rsi(NVDA_data, window=14)


#print(NVDA_data[['date', 'close', 'RSI']].head(15))


# MACD calculation formula
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    It is usually calculated for 12-day period in short window,26 day period in long_window.
    A 9 day EMA of the MACD, called the “signal line“, 
    """
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()

    macd = short_ema - long_ema

    # Calculation of signal line
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()

   # Calculation of histogram
    histogram = macd - signal_line

    return macd, signal_line, histogram,short_ema,long_ema


NVDA_data['MACD'], NVDA_data['Signal_Line'], NVDA_data['Histogram'], NVDA_data['short_ema'],NVDA_data['long_ema'] = calculate_macd(NVDA_data)

#print(NVDA_data[['date', 'close', 'MACD', 'Signal_Line', 'Histogram']].head(14))


# Calculation of Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    It is usually calculated for 20-day period window.
    Generally num_stand select 2.  
    """
    # Simple Moving Average 
    sma = data['close'].rolling(window=window, min_periods=1).mean()
    
    std = data['close'].rolling(window=window, min_periods=1).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    
     # B% Percentage position of price within Bollinger Bands
    bollinger_percent_b = (data['close'] - lower_band) / (upper_band - lower_band)

    # Volatility Measurement()
    bollinger_bandwidth = (upper_band - lower_band) / sma

    return bollinger_percent_b, bollinger_bandwidth,sma


NVDA_data['%B'], NVDA_data['Boolinger_Bandwidth'],NVDA_data['sma'] = calculate_bollinger_bands(NVDA_data)

# Calculation of MFI (Money Flow Index) 
def calculate_mfi(data, window=14):
    """
    It is usually calculated for 14-day period window.
    """
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    money_flow = typical_price * data['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    positive_mf = positive_flow.rolling(window=window, min_periods=1).sum()
    negative_mf = negative_flow.rolling(window=window, min_periods=1).sum()
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi

# Calculation OBV (On-Balance Volume) 
def calculate_obv(data):
    """
    OBV measures the relationship between price and volume. If the price is rising, volume is added, if it is falling, it is subtracted.   
    """
    # Obv initialize with 0.
    obv = [0]  

    for i in range(1, len(data)):
        if data['close'].iloc[i] > data['close'].iloc[i - 1]:
            obv.append(obv[-1] + data['volume'].iloc[i])  
            # Add volume if price rises
        elif data['close'].iloc[i] < data['close'].iloc[i - 1]:
            obv.append(obv[-1] - data['volume'].iloc[i])  
            #Subtract volume if price reduced.
        else:
            obv.append(obv[-1])  
            # If the price is not change,obs is not change

    return obv

NVDA_data['MFI'] = calculate_mfi(NVDA_data, window=14)
NVDA_data['OBV'] = calculate_obv(NVDA_data)

#Calculation of fibonacci values.
def calculate_fibonacci(data, column='close', window=14):
    
    # Max and min prices
    rolling_max = data[column].rolling(window=window, min_periods=3).max()
    rolling_min = data[column].rolling(window=window, min_periods=3).min()

    # Calculate the Fibonacci levels.
    fib_23_6 = rolling_max - 0.236 * (rolling_max - rolling_min)
    fib_38_2 = rolling_max - 0.382 * (rolling_max - rolling_min)
    fib_50_0 = rolling_max - 0.5 * (rolling_max - rolling_min)
    fib_61_8 = rolling_max - 0.618 * (rolling_max - rolling_min)

    return fib_23_6, fib_38_2, fib_50_0, fib_61_8



NVDA_data['fib_23_6'],NVDA_data['fib_38_2'],NVDA_data['fib_50_0'] ,NVDA_data['fib_61_8']  = calculate_fibonacci(NVDA_data)

#Calculation of Stochastic Oscillator.
def calculate_stochastic_oscillator(data, window=14):
    """
    It is usually calculated over a 14-day period 
    """

    # Determine high and low price
    high_max = data['high'].rolling(window=window, min_periods=1).max()
    low_min = data['low'].rolling(window=window, min_periods=1).min()    
    close_price = data['close']
    
    fast_stochastic = 100 * (close_price - low_min) / (high_max - low_min)
    
    stochastic_k = fast_stochastic
    stochastic_d = stochastic_k.rolling(window=3, min_periods=1).mean()  
    
    return stochastic_k, stochastic_d

NVDA_data['stochastic_k'], NVDA_data['stochastic_d'] = calculate_stochastic_oscillator(NVDA_data)


print(NVDA_data[['date',"volume", 'close', 'RSI','MACD',"sma","short_ema","long_ema", 'Signal_Line', 'Histogram','%B','Boolinger_Bandwidth',"MFI","OBV","fib_23_6","fib_38_2","fib_50_0","fib_61_8","stochastic_k","stochastic_d"]].head(30))
NVDA_data = NVDA_data[3:].copy()

#print(NVDA_data[['date',"volume", 'close', 'RSI','MACD',"sma","short_ema","long_ema", 'Signal_Line', 'Histogram',"ROC"]].head(30))
NVDA_data['date'] = pd.to_datetime(NVDA_data['date'], errors='coerce')

NVDA_data['year'] = NVDA_data['date'].dt.year
NVDA_data['month'] = NVDA_data['date'].dt.month
NVDA_data['day_of_week'] = NVDA_data['date'].dt.weekday
NVDA_data['day_of_year'] = NVDA_data['date'].dt.dayofyear


features = ['volume', 'RSI', 'MACD', 'sma', 'short_ema', 'long_ema', 'Signal_Line', 'Histogram',
            '%B', 'Boolinger_Bandwidth', 'MFI', 'OBV', 'fib_23_6', 'fib_38_2', 'fib_50_0', 
            'fib_61_8', 'stochastic_k', 'stochastic_d', 'year', 'month', 'day_of_week', 'day_of_year']
print(NVDA_data[features])
target = 'close'  # Target values


train_size = int(len(NVDA_data) * 0.8)

# Train/test 
train_data = NVDA_data.iloc[:train_size]
test_data = NVDA_data.iloc[train_size:]

X_train, y_train = train_data[features], train_data[target]
X_test, y_test = test_data[features], test_data[target]

# Check the train/test size
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

#Data scale
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_scaled_df = pd.DataFrame(X_train_scaled)
X_test_scaled_df = pd.DataFrame(X_test_scaled)
X_train_scaled_df.to_csv("/Users/ilknurakcay/Desktop/finance/X_train_scaled", index=False)
X_test_scaled_df.to_csv("/Users/ilknurakcay/Desktop/finance/X_test_scaled", index=False)
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)


target_scaler = MinMaxScaler(feature_range=(0, 1))
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = target_scaler.transform(y_test.values.reshape(-1, 1))

y_train_scaled_df = pd.DataFrame(y_train_scaled)
y_test_scaled_df = pd.DataFrame(y_test_scaled)
