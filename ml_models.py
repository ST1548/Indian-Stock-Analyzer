import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

def extract_features(data):
    """
    Extract features from technical indicators for ML models.
    
    Args:
        data (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    df = data.copy()
    
    # List of features to extract if available
    potential_features = [
        # Moving averages and price
        'SMA20', 'SMA50', 'SMA200', 'EMA9', 'EMA21',
        
        # Oscillators
        'RSI', 'MFI', 'CMO',
        
        # MACD
        'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9',
        
        # Bollinger Bands
        'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'BBB_20_2.0', 'BBP_20_2.0',
        
        # Stochastic
        'STOCHk_14_3_3', 'STOCHd_14_3_3',
        
        # ADX
        'ADX_14', 'DMP_14', 'DMN_14',
        
        # Volume indicators
        'OBV', 'Volume_SMA20',
        
        # ATR
        'ATR'
    ]
    
    # Check which features are available
    available_features = [feature for feature in potential_features if feature in df.columns]
    
    # Create basic features
    features = pd.DataFrame(index=df.index)
    
    # Add relative price to moving averages if available
    if 'SMA20' in df.columns:
        features['price_to_sma20'] = df['Close'] / df['SMA20']
    if 'SMA50' in df.columns:
        features['price_to_sma50'] = df['Close'] / df['SMA50']
    if 'SMA200' in df.columns:
        features['price_to_sma200'] = df['Close'] / df['SMA200']
    
    # Add moving average crossovers if available
    if 'SMA20' in df.columns and 'SMA50' in df.columns:
        features['sma20_to_sma50'] = df['SMA20'] / df['SMA50']
    if 'EMA9' in df.columns and 'EMA21' in df.columns:
        features['ema9_to_ema21'] = df['EMA9'] / df['EMA21']
    
    # Add oscillator values directly
    for oscillator in ['RSI', 'MFI', 'CMO']:
        if oscillator in df.columns:
            features[oscillator] = df[oscillator]
    
    # Add MACD features if available
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        features['macd_hist'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        features['macd_signal_diff'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    # Add Bollinger Band features if available
    if 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns and 'BBL_20_2.0' in df.columns:
        features['bb_width'] = (df['BBU_20_2.0'] - df['BBL_20_2.0']) / df['BBM_20_2.0']
        features['bb_position'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
    
    # Add Stochastic features if available
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
        features['stoch_k'] = df['STOCHk_14_3_3']
        features['stoch_d'] = df['STOCHd_14_3_3']
        features['stoch_k_d_diff'] = df['STOCHk_14_3_3'] - df['STOCHd_14_3_3']
    
    # Add ADX features if available
    if 'ADX_14' in df.columns and 'DMP_14' in df.columns and 'DMN_14' in df.columns:
        features['adx'] = df['ADX_14']
        features['dmi_diff'] = df['DMP_14'] - df['DMN_14']
        features['dmi_ratio'] = df['DMP_14'] / df['DMN_14'] if (df['DMN_14'] != 0).all() else 1.0
    
    # Add volume features if available
    if 'Volume' in df.columns and 'Volume_SMA20' in df.columns:
        features['volume_ratio'] = df['Volume'] / df['Volume_SMA20']
    
    # Add price momentum features
    features['price_change_1d'] = df['Close'].pct_change(1)
    features['price_change_5d'] = df['Close'].pct_change(5)
    
    # Add volatility feature if ATR is available
    if 'ATR' in df.columns:
        features['atr_ratio'] = df['ATR'] / df['Close']
    
    # Handle NaN values
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(method='ffill', inplace=True)
    features.fillna(0, inplace=True)
    
    return features

def generate_labels(data, prediction_horizon=5):
    """
    Generate classification labels for price movement prediction.
    
    Args:
        data (pd.DataFrame): DataFrame with stock price data
        prediction_horizon (int): Number of periods ahead to predict
        
    Returns:
        pd.Series: Labels for price movement (1 for up, 0 for down)
    """
    # Calculate future returns
    future_returns = data['Close'].shift(-prediction_horizon) / data['Close'] - 1
    
    # Create binary labels (1 for positive return, 0 for negative)
    labels = (future_returns > 0).astype(int)
    
    return labels

def train_model(features, labels, model_type='random_forest'):
    """
    Train a machine learning model for price prediction.
    
    Args:
        features (pd.DataFrame): Feature DataFrame
        labels (pd.Series): Target labels
        model_type (str): Type of model to train ('random_forest' or 'gradient_boosting')
        
    Returns:
        tuple: Trained model and scaler
    """
    # Create a scaler
    scaler = StandardScaler()
    
    # Drop rows with NaN values
    features = features.dropna()
    labels = labels.loc[features.index]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Scale the features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the model
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=None,
            min_samples_split=10,
            random_state=42
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    
    return model, scaler

def predict_stock_movement(data, ticker, trading_style='swing'):
    """
    Predict stock movement and generate trading recommendations.
    
    Args:
        data (pd.DataFrame): DataFrame with technical indicators
        ticker (str): Stock ticker symbol
        trading_style (str): 'intraday' or 'swing'
        
    Returns:
        dict: Dictionary with prediction results
    """
    # Determine prediction horizon based on trading style
    prediction_horizon = 1 if trading_style == 'intraday' else 5
    
    # Since we don't have actual historical data for accurate model training,
    # we'll simulate a prediction based on technical indicator signals
    
    # Get features
    features = extract_features(data)
    
    # Get the latest data point for prediction
    latest_features = features.iloc[-1:] if not features.empty else None
    
    # Create a simulated prediction (in a real implementation, this would use a trained model)
    # For demonstration, we'll use a weighted ensemble of technical signals
    
    # Get the technical signal scores
    ma_score = 0
    if 'price_to_sma20' in features.columns:
        ma_score += 1 if latest_features['price_to_sma20'].iloc[0] > 1 else -1
    if 'price_to_sma50' in features.columns:
        ma_score += 1 if latest_features['price_to_sma50'].iloc[0] > 1 else -1
    if 'sma20_to_sma50' in features.columns:
        ma_score += 1 if latest_features['sma20_to_sma50'].iloc[0] > 1 else -1
    
    rsi_score = 0
    if 'RSI' in features.columns:
        rsi = latest_features['RSI'].iloc[0]
        if rsi < 30:
            rsi_score += 1  # Oversold - bullish
        elif rsi > 70:
            rsi_score -= 1  # Overbought - bearish
        elif rsi > 50:
            rsi_score += 0.5  # Above 50 - mildly bullish
        else:
            rsi_score -= 0.5  # Below 50 - mildly bearish
    
    macd_score = 0
    if 'macd_hist' in features.columns:
        macd_hist = latest_features['macd_hist'].iloc[0]
        if macd_hist > 0:
            macd_score += 0.5  # Positive histogram - bullish
        else:
            macd_score -= 0.5  # Negative histogram - bearish
    if 'macd_signal_diff' in features.columns:
        macd_diff = latest_features['macd_signal_diff'].iloc[0]
        if macd_diff > 0:
            macd_score += 0.5  # MACD above signal - bullish
        else:
            macd_score -= 0.5  # MACD below signal - bearish
    
    bb_score = 0
    if 'bb_position' in features.columns:
        bb_pos = latest_features['bb_position'].iloc[0]
        if bb_pos < 0.2:
            bb_score += 1  # Near lower band - bullish
        elif bb_pos > 0.8:
            bb_score -= 1  # Near upper band - bearish
    
    stoch_score = 0
    if 'stoch_k' in features.columns and 'stoch_d' in features.columns:
        stoch_k = latest_features['stoch_k'].iloc[0]
        stoch_d = latest_features['stoch_d'].iloc[0]
        if stoch_k < 20 and stoch_d < 20:
            stoch_score += 1  # Oversold - bullish
        elif stoch_k > 80 and stoch_d > 80:
            stoch_score -= 1  # Overbought - bearish
        if stoch_k > stoch_d:
            stoch_score += 0.5  # K above D - bullish
        else:
            stoch_score -= 0.5  # K below D - bearish
    
    adx_score = 0
    if 'adx' in features.columns and 'dmi_diff' in features.columns:
        adx = latest_features['adx'].iloc[0]
        dmi_diff = latest_features['dmi_diff'].iloc[0]
        if adx > 25 and dmi_diff > 0:
            adx_score += 1  # Strong trend and +DMI > -DMI - bullish
        elif adx > 25 and dmi_diff < 0:
            adx_score -= 1  # Strong trend and +DMI < -DMI - bearish
    
    volume_score = 0
    if 'volume_ratio' in features.columns:
        volume_ratio = latest_features['volume_ratio'].iloc[0]
        if volume_ratio > 1.5:
            # Check if price is up or down
            price_change = latest_features['price_change_1d'].iloc[0]
            if price_change > 0:
                volume_score += 1  # High volume up day - bullish
            else:
                volume_score -= 1  # High volume down day - bearish
    
    # Adjust weights based on trading style
    if trading_style == 'intraday':
        # For intraday, give more weight to oscillators and less to moving averages
        total_score = (
            ma_score * 0.5 +
            rsi_score * 1.2 +
            macd_score * 0.8 +
            bb_score * 1.0 +
            stoch_score * 1.2 +
            adx_score * 0.8 +
            volume_score * 1.5
        )
    else:  # swing
        # For swing trading, give more weight to trend indicators
        total_score = (
            ma_score * 1.5 +
            rsi_score * 0.8 +
            macd_score * 1.2 +
            bb_score * 0.7 +
            stoch_score * 0.8 +
            adx_score * 1.2 +
            volume_score * 0.8
        )
    
    # Determine prediction and confidence
    if total_score > 2:
        prediction = 'buy'
        confidence = min(0.9, 0.5 + abs(total_score) / 10)
    elif total_score > 0:
        prediction = 'buy'
        confidence = 0.5 + abs(total_score) / 10
    elif total_score < -2:
        prediction = 'sell'
        confidence = min(0.9, 0.5 + abs(total_score) / 10)
    elif total_score < 0:
        prediction = 'sell'
        confidence = 0.5 + abs(total_score) / 10
    else:
        prediction = 'neutral'
        confidence = 0.5
    
    # Generate reasoning based on scores
    reasoning_points = []
    
    if ma_score > 0:
        reasoning_points.append("• Price is above key moving averages, indicating bullish trend.")
    elif ma_score < 0:
        reasoning_points.append("• Price is below key moving averages, indicating bearish trend.")
        
    if rsi_score > 0:
        if 'RSI' in features.columns and latest_features['RSI'].iloc[0] < 30:
            reasoning_points.append(f"• RSI is in oversold territory ({latest_features['RSI'].iloc[0]:.1f}), suggesting potential upward reversal.")
        else:
            reasoning_points.append("• RSI indicates positive momentum.")
    elif rsi_score < 0:
        if 'RSI' in features.columns and latest_features['RSI'].iloc[0] > 70:
            reasoning_points.append(f"• RSI is in overbought territory ({latest_features['RSI'].iloc[0]:.1f}), suggesting potential downward reversal.")
        else:
            reasoning_points.append("• RSI indicates negative momentum.")
    
    if macd_score > 0:
        reasoning_points.append("• MACD shows bullish signal.")
    elif macd_score < 0:
        reasoning_points.append("• MACD shows bearish signal.")
    
    if bb_score > 0:
        reasoning_points.append("• Price near lower Bollinger Band, suggesting potential buy opportunity.")
    elif bb_score < 0:
        reasoning_points.append("• Price near upper Bollinger Band, suggesting potential sell opportunity.")
    
    if stoch_score > 0:
        reasoning_points.append("• Stochastic oscillator indicates bullish momentum.")
    elif stoch_score < 0:
        reasoning_points.append("• Stochastic oscillator indicates bearish momentum.")
    
    if adx_score > 0:
        reasoning_points.append("• ADX suggests strong uptrend in progress.")
    elif adx_score < 0:
        reasoning_points.append("• ADX suggests strong downtrend in progress.")
    
    if volume_score > 0:
        reasoning_points.append("• Volume analysis shows strong buying interest.")
    elif volume_score < 0:
        reasoning_points.append("• Volume analysis shows strong selling pressure.")
    
    # Add a timeframe-specific note
    if trading_style == 'intraday':
        reasoning_points.append(f"• This analysis is optimized for intraday trading over the next few hours.")
    else:
        reasoning_points.append(f"• This analysis is optimized for swing trading over the next several days.")
    
    # Combine reasoning points
    reasoning = "\n".join(reasoning_points)
    
    results = {
        'recommendation': prediction,
        'confidence': confidence,
        'reasoning': reasoning,
        'trading_style': trading_style,
        'underlying_scores': {
            'moving_averages': ma_score,
            'rsi': rsi_score,
            'macd': macd_score,
            'bollinger_bands': bb_score,
            'stochastic': stoch_score,
            'adx': adx_score,
            'volume': volume_score,
            'total': total_score
        }
    }
    
    return results
