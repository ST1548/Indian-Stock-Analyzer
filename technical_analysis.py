import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta

def calculate_technical_indicators(stock_data):
    """
    Calculates technical indicators for a given stock data.
    
    Args:
        stock_data (pd.DataFrame): DataFrame containing stock OHLCV data
        
    Returns:
        pd.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = stock_data.copy()
    
    # Ensure we have OHLCV columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Stock data must contain OHLCV columns")
    
    # Moving Averages
    df['SMA20'] = ta.sma(df['Close'], length=20)
    df['SMA50'] = ta.sma(df['Close'], length=50)
    df['SMA200'] = ta.sma(df['Close'], length=200)
    df['EMA9'] = ta.ema(df['Close'], length=9)
    df['EMA21'] = ta.ema(df['Close'], length=21)
    
    # Relative Strength Index (RSI)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    
    # Moving Average Convergence Divergence (MACD)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    
    # Bollinger Bands
    bbands = ta.bbands(df['Close'], length=20, std=2)
    df = pd.concat([df, bbands], axis=1)
    
    # Stochastic Oscillator
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    df = pd.concat([df, stoch], axis=1)
    
    # Average Directional Index (ADX)
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df = pd.concat([df, adx], axis=1)
    
    # On-Balance Volume (OBV)
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    
    # Volume Moving Average
    df['Volume_SMA20'] = ta.sma(df['Volume'], length=20)
    
    # Average True Range (ATR)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # Percentage Price Oscillator (PPO)
    ppo = ta.ppo(df['Close'], fast=12, slow=26, signal=9)
    df = pd.concat([df, ppo], axis=1)
    
    # Ichimoku Cloud
    try:
        ichimoku = ta.ichimoku(df['High'], df['Low'], df['Close'])
        df = pd.concat([df, ichimoku], axis=1)
    except:
        # If Ichimoku fails due to not enough data, continue without it
        pass
    
    # Chande Momentum Oscillator (CMO)
    df['CMO'] = ta.cmo(df['Close'], length=14)
    
    # Money Flow Index (MFI)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    
    # Remove NaN values for indicators that require historical data
    df = df.replace([np.inf, -np.inf], np.nan)
    
    return df

def generate_technical_signals(df):
    """
    Generates trading signals and explanations based on technical indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with technical indicators
        
    Returns:
        dict: Dictionary containing signals and explanations
    """
    signals = {}
    
    # Moving Average signals
    ma_signals = []
    ma_score = 0
    
    # SMA20 and SMA50 crossover
    if 'SMA20' in df.columns and 'SMA50' in df.columns:
        if df['SMA20'].iloc[-2] < df['SMA50'].iloc[-2] and df['SMA20'].iloc[-1] > df['SMA50'].iloc[-1]:
            ma_signals.append("➕ **Golden Cross (SMA20 crossed above SMA50)**: Bullish signal indicating potential uptrend.")
            ma_score += 1
        elif df['SMA20'].iloc[-2] > df['SMA50'].iloc[-2] and df['SMA20'].iloc[-1] < df['SMA50'].iloc[-1]:
            ma_signals.append("➖ **Death Cross (SMA20 crossed below SMA50)**: Bearish signal indicating potential downtrend.")
            ma_score -= 1
    
    # Price above/below SMAs
    if 'SMA20' in df.columns:
        if df['Close'].iloc[-1] > df['SMA20'].iloc[-1]:
            ma_signals.append("➕ **Price above SMA20**: Short-term bullish indication.")
            ma_score += 0.5
        else:
            ma_signals.append("➖ **Price below SMA20**: Short-term bearish indication.")
            ma_score -= 0.5
    
    if 'SMA50' in df.columns:
        if df['Close'].iloc[-1] > df['SMA50'].iloc[-1]:
            ma_signals.append("➕ **Price above SMA50**: Medium-term bullish indication.")
            ma_score += 0.5
        else:
            ma_signals.append("➖ **Price below SMA50**: Medium-term bearish indication.")
            ma_score -= 0.5
    
    if 'SMA200' in df.columns:
        if df['Close'].iloc[-1] > df['SMA200'].iloc[-1]:
            ma_signals.append("➕ **Price above SMA200**: Long-term bullish indication.")
            ma_score += 0.5
        else:
            ma_signals.append("➖ **Price below SMA200**: Long-term bearish indication.")
            ma_score -= 0.5
    
    # EMA crossovers
    if 'EMA9' in df.columns and 'EMA21' in df.columns:
        if df['EMA9'].iloc[-2] < df['EMA21'].iloc[-2] and df['EMA9'].iloc[-1] > df['EMA21'].iloc[-1]:
            ma_signals.append("➕ **EMA9 crossed above EMA21**: Bullish signal for short-term momentum.")
            ma_score += 1
        elif df['EMA9'].iloc[-2] > df['EMA21'].iloc[-2] and df['EMA9'].iloc[-1] < df['EMA21'].iloc[-1]:
            ma_signals.append("➖ **EMA9 crossed below EMA21**: Bearish signal for short-term momentum.")
            ma_score -= 1
    
    signals['moving_averages'] = 'bullish' if ma_score > 0 else 'bearish' if ma_score < 0 else 'neutral'
    signals['moving_averages_score'] = ma_score
    signals['moving_averages_explanation'] = "\n".join(ma_signals) if ma_signals else "No clear moving average signals."
    
    # RSI signals
    rsi_signals = []
    rsi_score = 0
    
    if 'RSI' in df.columns:
        current_rsi = df['RSI'].iloc[-1]
        previous_rsi = df['RSI'].iloc[-2]
        
        if current_rsi < 30:
            rsi_signals.append(f"➕ **RSI is oversold ({current_rsi:.2f})**: Potential buying opportunity.")
            rsi_score += 1
        elif current_rsi > 70:
            rsi_signals.append(f"➖ **RSI is overbought ({current_rsi:.2f})**: Potential selling opportunity.")
            rsi_score -= 1
        
        # RSI divergence
        if len(df) >= 5:  # Need enough data points to check for divergence
            if (df['Close'].iloc[-1] > df['Close'].iloc[-5]) and (df['RSI'].iloc[-1] < df['RSI'].iloc[-5]):
                rsi_signals.append("➖ **Bearish RSI divergence**: Price making higher highs while RSI making lower highs.")
                rsi_score -= 1
            elif (df['Close'].iloc[-1] < df['Close'].iloc[-5]) and (df['RSI'].iloc[-1] > df['RSI'].iloc[-5]):
                rsi_signals.append("➕ **Bullish RSI divergence**: Price making lower lows while RSI making higher lows.")
                rsi_score += 1
        
        # RSI trend
        if previous_rsi < 50 and current_rsi > 50:
            rsi_signals.append("➕ **RSI crossed above 50**: Bullish momentum building.")
            rsi_score += 0.5
        elif previous_rsi > 50 and current_rsi < 50:
            rsi_signals.append("➖ **RSI crossed below 50**: Bearish momentum building.")
            rsi_score -= 0.5
    
    signals['rsi'] = 'bullish' if rsi_score > 0 else 'bearish' if rsi_score < 0 else 'neutral'
    signals['rsi_score'] = rsi_score
    signals['rsi_explanation'] = "\n".join(rsi_signals) if rsi_signals else "RSI is in neutral territory."
    
    # MACD signals
    macd_signals = []
    macd_score = 0
    
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        macd_line = df['MACD_12_26_9'].iloc[-1]
        signal_line = df['MACDs_12_26_9'].iloc[-1]
        prev_macd_line = df['MACD_12_26_9'].iloc[-2]
        prev_signal_line = df['MACDs_12_26_9'].iloc[-2]
        
        # MACD crossover
        if prev_macd_line < prev_signal_line and macd_line > signal_line:
            macd_signals.append("➕ **MACD crossed above signal line**: Bullish signal.")
            macd_score += 1
        elif prev_macd_line > prev_signal_line and macd_line < signal_line:
            macd_signals.append("➖ **MACD crossed below signal line**: Bearish signal.")
            macd_score -= 1
        
        # MACD histogram
        if 'MACDh_12_26_9' in df.columns:
            hist = df['MACDh_12_26_9'].iloc[-1]
            prev_hist = df['MACDh_12_26_9'].iloc[-2]
            
            if hist > 0 and prev_hist < 0:
                macd_signals.append("➕ **MACD histogram turned positive**: Bullish momentum building.")
                macd_score += 0.5
            elif hist < 0 and prev_hist > 0:
                macd_signals.append("➖ **MACD histogram turned negative**: Bearish momentum building.")
                macd_score -= 0.5
            elif hist > 0 and hist > prev_hist:
                macd_signals.append("➕ **MACD histogram increasing**: Bullish momentum strengthening.")
                macd_score += 0.25
            elif hist < 0 and hist < prev_hist:
                macd_signals.append("➖ **MACD histogram decreasing**: Bearish momentum strengthening.")
                macd_score -= 0.25
        
        # MACD zero line
        if (prev_macd_line < 0 and macd_line > 0):
            macd_signals.append("➕ **MACD crossed above zero line**: Bullish trend confirmation.")
            macd_score += 0.5
        elif (prev_macd_line > 0 and macd_line < 0):
            macd_signals.append("➖ **MACD crossed below zero line**: Bearish trend confirmation.")
            macd_score -= 0.5
    
    signals['macd'] = 'bullish' if macd_score > 0 else 'bearish' if macd_score < 0 else 'neutral'
    signals['macd_score'] = macd_score
    signals['macd_explanation'] = "\n".join(macd_signals) if macd_signals else "No clear MACD signals."
    
    # Bollinger Bands signals
    bb_signals = []
    bb_score = 0
    
    if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
        lower_band = df['BBL_20_2.0'].iloc[-1]
        middle_band = df['BBM_20_2.0'].iloc[-1]
        upper_band = df['BBU_20_2.0'].iloc[-1]
        close_price = df['Close'].iloc[-1]
        
        # Price near bands
        if close_price <= lower_band:
            bb_signals.append("➕ **Price at/below lower Bollinger Band**: Potential oversold condition, watch for reversal.")
            bb_score += 0.75
        elif close_price >= upper_band:
            bb_signals.append("➖ **Price at/above upper Bollinger Band**: Potential overbought condition, watch for reversal.")
            bb_score -= 0.75
        
        # Band width - volatility
        if 'BBB_20_2.0' in df.columns:
            band_width = df['BBB_20_2.0'].iloc[-1]
            prev_band_width = df['BBB_20_2.0'].iloc[-5] if len(df) > 5 else band_width
            
            if band_width < prev_band_width * 0.8:
                bb_signals.append("📊 **Bollinger Band squeeze**: Decreasing volatility, potential for breakout.")
            elif band_width > prev_band_width * 1.2:
                bb_signals.append("📊 **Bollinger Bands widening**: Increasing volatility, trend likely to continue.")
        
        # Bollinger Band bounce
        if len(df) > 2:
            prev_close = df['Close'].iloc[-2]
            if prev_close <= lower_band and close_price > prev_close:
                bb_signals.append("➕ **Bounce from lower Bollinger Band**: Potential bullish reversal.")
                bb_score += 1
            elif prev_close >= upper_band and close_price < prev_close:
                bb_signals.append("➖ **Bounce from upper Bollinger Band**: Potential bearish reversal.")
                bb_score -= 1
    
    signals['bollinger'] = 'bullish' if bb_score > 0 else 'bearish' if bb_score < 0 else 'neutral'
    signals['bollinger_score'] = bb_score
    signals['bollinger_explanation'] = "\n".join(bb_signals) if bb_signals else "Price is within normal Bollinger Bands range."
    
    # Stochastic signals
    stoch_signals = []
    stoch_score = 0
    
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
        k_line = df['STOCHk_14_3_3'].iloc[-1]
        d_line = df['STOCHd_14_3_3'].iloc[-1]
        prev_k_line = df['STOCHk_14_3_3'].iloc[-2]
        prev_d_line = df['STOCHd_14_3_3'].iloc[-2]
        
        # Overbought/oversold
        if k_line < 20 and d_line < 20:
            stoch_signals.append(f"➕ **Stochastic in oversold territory (K: {k_line:.2f}, D: {d_line:.2f})**: Potential buying opportunity.")
            stoch_score += 0.75
        elif k_line > 80 and d_line > 80:
            stoch_signals.append(f"➖ **Stochastic in overbought territory (K: {k_line:.2f}, D: {d_line:.2f})**: Potential selling opportunity.")
            stoch_score -= 0.75
        
        # Stochastic crossover
        if prev_k_line < prev_d_line and k_line > d_line:
            stoch_signals.append("➕ **Stochastic %K crossed above %D**: Bullish signal.")
            stoch_score += 1
        elif prev_k_line > prev_d_line and k_line < d_line:
            stoch_signals.append("➖ **Stochastic %K crossed below %D**: Bearish signal.")
            stoch_score -= 1
        
        # Divergence
        if len(df) >= 5:
            if (df['Close'].iloc[-1] > df['Close'].iloc[-5]) and (k_line < df['STOCHk_14_3_3'].iloc[-5]):
                stoch_signals.append("➖ **Bearish Stochastic divergence**: Price making higher highs while Stochastic making lower highs.")
                stoch_score -= 0.5
            elif (df['Close'].iloc[-1] < df['Close'].iloc[-5]) and (k_line > df['STOCHk_14_3_3'].iloc[-5]):
                stoch_signals.append("➕ **Bullish Stochastic divergence**: Price making lower lows while Stochastic making higher lows.")
                stoch_score += 0.5
    
    signals['stochastic'] = 'bullish' if stoch_score > 0 else 'bearish' if stoch_score < 0 else 'neutral'
    signals['stochastic_score'] = stoch_score
    signals['stochastic_explanation'] = "\n".join(stoch_signals) if stoch_signals else "No clear Stochastic Oscillator signals."
    
    # Volume signals
    volume_signals = []
    volume_score = 0
    
    if 'Volume' in df.columns:
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume'].iloc[-20:].mean() if len(df) >= 20 else df['Volume'].mean()
        
        # Volume spike
        if current_volume > avg_volume * 1.5:
            # Check if price increased or decreased
            if df['Close'].iloc[-1] > df['Open'].iloc[-1]:
                volume_signals.append(f"➕ **High volume up day (Volume: {current_volume:,.0f})**: Strong bullish interest.")
                volume_score += 1
            else:
                volume_signals.append(f"➖ **High volume down day (Volume: {current_volume:,.0f})**: Strong bearish interest.")
                volume_score -= 1
                
        # Low volume
        elif current_volume < avg_volume * 0.5:
            volume_signals.append(f"⚠️ **Low volume (Volume: {current_volume:,.0f})**: Lack of conviction in price move.")
        
        # Volume trend
        if 'Volume_SMA20' in df.columns:
            vol_sma = df['Volume_SMA20'].iloc[-1]
            if current_volume > vol_sma:
                volume_signals.append("📊 **Volume above average**: Increased interest in the stock.")
            else:
                volume_signals.append("📊 **Volume below average**: Decreased interest in the stock.")
        
        # On-Balance Volume (OBV)
        if 'OBV' in df.columns and len(df) >= 5:
            current_obv = df['OBV'].iloc[-1]
            prev_obv = df['OBV'].iloc[-5]
            
            if current_obv > prev_obv and df['Close'].iloc[-1] > df['Close'].iloc[-5]:
                volume_signals.append("➕ **Rising OBV with rising price**: Confirmation of uptrend.")
                volume_score += 0.5
            elif current_obv < prev_obv and df['Close'].iloc[-1] < df['Close'].iloc[-5]:
                volume_signals.append("➖ **Falling OBV with falling price**: Confirmation of downtrend.")
                volume_score -= 0.5
            elif current_obv > prev_obv and df['Close'].iloc[-1] < df['Close'].iloc[-5]:
                volume_signals.append("➕ **Rising OBV with falling price**: Potential bullish divergence.")
                volume_score += 0.75
            elif current_obv < prev_obv and df['Close'].iloc[-1] > df['Close'].iloc[-5]:
                volume_signals.append("➖ **Falling OBV with rising price**: Potential bearish divergence.")
                volume_score -= 0.75
    
    signals['volume'] = 'bullish' if volume_score > 0 else 'bearish' if volume_score < 0 else 'neutral'
    signals['volume_score'] = volume_score
    signals['volume_explanation'] = "\n".join(volume_signals) if volume_signals else "No significant volume signals."
    
    # Combine all signal scores to get an overall technical score
    signal_types = ['moving_averages', 'rsi', 'macd', 'bollinger', 'stochastic', 'volume']
    total_score = sum(signals.get(f'{signal_type}_score', 0) for signal_type in signal_types)
    
    # Determine overall signal
    if total_score >= 2:
        signals['overall'] = 'strong_buy'
    elif total_score > 0:
        signals['overall'] = 'buy'
    elif total_score == 0:
        signals['overall'] = 'neutral'
    elif total_score > -2:
        signals['overall'] = 'sell'
    else:
        signals['overall'] = 'strong_sell'
    
    signals['overall_score'] = total_score
    
    return signals
