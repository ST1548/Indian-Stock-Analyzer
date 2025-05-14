import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Custom import for pandas_ta that handles the pkg_resources issue
try:
    import pandas_ta as ta
except ImportError:
    # If importing pandas_ta fails due to pkg_resources, create our own minimal version
    # with the functions we need for this application
    class MinimalTA:
        @staticmethod
        def sma(series, length=None):
            return series.rolling(window=length).mean()
        
        @staticmethod
        def ema(series, length=None):
            return series.ewm(span=length, adjust=False).mean()
        
        @staticmethod
        def rsi(series, length=None):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        @staticmethod
        def macd(series, fast=12, slow=26, signal=9):
            ema_fast = series.ewm(span=fast, adjust=False).mean()
            ema_slow = series.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            histogram = macd_line - signal_line
            
            result = pd.DataFrame()
            result[f'MACD_{fast}_{slow}_{signal}'] = macd_line
            result[f'MACDs_{fast}_{slow}_{signal}'] = signal_line
            result[f'MACDh_{fast}_{slow}_{signal}'] = histogram
            return result
        
        @staticmethod
        def bbands(series, length=20, std=2):
            middle_band = series.rolling(window=length).mean()
            std_dev = series.rolling(window=length).std()
            upper_band = middle_band + std * std_dev
            lower_band = middle_band - std * std_dev
            bandwidth = (upper_band - lower_band) / middle_band
            
            result = pd.DataFrame()
            result[f'BBL_{length}_{std}.0'] = lower_band
            result[f'BBM_{length}_{std}.0'] = middle_band
            result[f'BBU_{length}_{std}.0'] = upper_band
            result[f'BBB_{length}_{std}.0'] = bandwidth
            return result
        
        @staticmethod
        def stoch(high, low, close, k=14, d=3, smooth_k=3):
            lowest_low = low.rolling(window=k).min()
            highest_high = high.rolling(window=k).max()
            stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            stoch_k_smoothed = stoch_k.rolling(window=smooth_k).mean()
            stoch_d = stoch_k_smoothed.rolling(window=d).mean()
            
            result = pd.DataFrame()
            result[f'STOCHk_{k}_{d}_{smooth_k}'] = stoch_k_smoothed
            result[f'STOCHd_{k}_{d}_{smooth_k}'] = stoch_d
            return result
            
        @staticmethod
        def adx(high, low, close, length=14):
            tr1 = abs(high - low)
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            atr = tr.rolling(window=length).mean()
            
            # Plus and minus directional movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            plus_di = 100 * (plus_dm.rolling(window=length).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=length).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx_val = dx.rolling(window=length).mean()
            
            result = pd.DataFrame()
            result[f'ADX_{length}'] = adx_val
            result[f'DMP_{length}'] = plus_di
            result[f'DMN_{length}'] = minus_di
            return result
        
        @staticmethod
        def obv(close, volume):
            return (np.sign(close.diff()) * volume).fillna(0).cumsum()
        
        @staticmethod
        def atr(high, low, close, length=14):
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            return tr.rolling(window=length).mean()
        
        @staticmethod
        def ppo(close, fast=12, slow=26, signal=9):
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            ppo_line = 100 * ((ema_fast - ema_slow) / ema_slow)
            ppo_signal = ppo_line.ewm(span=signal, adjust=False).mean()
            ppo_hist = ppo_line - ppo_signal
            
            result = pd.DataFrame()
            result[f'PPO_{fast}_{slow}_{signal}'] = ppo_line
            result[f'PPOs_{fast}_{slow}_{signal}'] = ppo_signal
            result[f'PPOh_{fast}_{slow}_{signal}'] = ppo_hist
            return result
        
        @staticmethod
        def ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
            # Simply return empty dataframe as this is a complex indicator
            # and rarely used in basic analysis
            return pd.DataFrame()
        
        @staticmethod
        def cmo(close, length=14):
            # Chande Momentum Oscillator
            delta = close.diff()
            up = delta.where(delta > 0, 0).rolling(window=length).sum()
            down = -delta.where(delta < 0, 0).rolling(window=length).sum()
            return 100 * ((up - down) / (up + down))
        
        @staticmethod
        def mfi(high, low, close, volume, length=14):
            # Money Flow Index
            typical_price = (high + low + close) / 3
            raw_money_flow = typical_price * volume
            
            delta = typical_price.diff()
            positive_flow = raw_money_flow.where(delta > 0, 0).rolling(window=length).sum()
            negative_flow = raw_money_flow.where(delta < 0, 0).rolling(window=length).sum()
            
            money_ratio = positive_flow / negative_flow
            return 100 - (100 / (1 + money_ratio))
    
    # Use our minimal implementation instead
    ta = MinimalTA()

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
    try:
        # Create a copy of the dataframe to avoid warnings
        mfi_data = pd.DataFrame()
        mfi_data['high'] = df['High'].astype(float)
        mfi_data['low'] = df['Low'].astype(float)
        mfi_data['close'] = df['Close'].astype(float)
        mfi_data['volume'] = df['Volume'].astype(float)
        
        # Calculate MFI using the clean data
        df['MFI'] = ta.mfi(
            high=mfi_data['high'], 
            low=mfi_data['low'], 
            close=mfi_data['close'], 
            volume=mfi_data['volume'], 
            length=14
        )
    except Exception as e:
        print(f"Warning: Could not calculate MFI: {e}")
        df['MFI'] = np.nan
    
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
    # Helper function to safely check if a value is a number
    def is_valid_number(value):
        """Check if a value is a valid number (not None, NaN, or infinite)"""
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return not (np.isnan(value) or np.isinf(value))
        return False
    
    # Helper function for safe comparison
    def safe_compare(a, b, comparison='greater'):
        """Safely compare two values that might be None or NaN"""
        if not is_valid_number(a) or not is_valid_number(b):
            return False
        
        if comparison == 'greater':
            return float(a) > float(b)
        elif comparison == 'less':
            return float(a) < float(b)
        elif comparison == 'equal':
            return float(a) == float(b)
        return False
    
    # Helper to get value safely
    def safe_get(df, column, index):
        """Safely get a value from the dataframe that might be None or NaN"""
        try:
            value = df[column].iloc[index]
            if is_valid_number(value):
                return value
            return None
        except:
            return None
            
    signals = {}
    
    # Moving Average signals
    ma_signals = []
    ma_score = 0
    
    # SMA20 and SMA50 crossover
    if 'SMA20' in df.columns and 'SMA50' in df.columns:
        sma20_prev = safe_get(df, 'SMA20', -2)
        sma50_prev = safe_get(df, 'SMA50', -2)
        sma20_curr = safe_get(df, 'SMA20', -1)
        sma50_curr = safe_get(df, 'SMA50', -1)
        
        if (sma20_prev is not None and sma50_prev is not None and 
            sma20_curr is not None and sma50_curr is not None):
            if safe_compare(sma20_prev, sma50_prev, 'less') and safe_compare(sma20_curr, sma50_curr, 'greater'):
                ma_signals.append("➕ **Golden Cross (SMA20 crossed above SMA50)**: Bullish signal indicating potential uptrend.")
                ma_score += 1
            elif safe_compare(sma20_prev, sma50_prev, 'greater') and safe_compare(sma20_curr, sma50_curr, 'less'):
                ma_signals.append("➖ **Death Cross (SMA20 crossed below SMA50)**: Bearish signal indicating potential downtrend.")
                ma_score -= 1
    
    # Price above/below SMAs
    if 'SMA20' in df.columns:
        close_curr = safe_get(df, 'Close', -1)
        sma20_curr = safe_get(df, 'SMA20', -1)
        
        if close_curr is not None and sma20_curr is not None:
            if safe_compare(close_curr, sma20_curr, 'greater'):
                ma_signals.append("➕ **Price above SMA20**: Short-term bullish indication.")
                ma_score += 0.5
            else:
                ma_signals.append("➖ **Price below SMA20**: Short-term bearish indication.")
                ma_score -= 0.5
    
    if 'SMA50' in df.columns:
        close_curr = safe_get(df, 'Close', -1)
        sma50_curr = safe_get(df, 'SMA50', -1)
        
        if close_curr is not None and sma50_curr is not None:
            if safe_compare(close_curr, sma50_curr, 'greater'):
                ma_signals.append("➕ **Price above SMA50**: Medium-term bullish indication.")
                ma_score += 0.5
            else:
                ma_signals.append("➖ **Price below SMA50**: Medium-term bearish indication.")
                ma_score -= 0.5
    
    if 'SMA200' in df.columns:
        close_curr = safe_get(df, 'Close', -1)
        sma200_curr = safe_get(df, 'SMA200', -1)
        
        if close_curr is not None and sma200_curr is not None:
            if safe_compare(close_curr, sma200_curr, 'greater'):
                ma_signals.append("➕ **Price above SMA200**: Long-term bullish indication.")
                ma_score += 0.5
            else:
                ma_signals.append("➖ **Price below SMA200**: Long-term bearish indication.")
                ma_score -= 0.5
    
    # EMA crossovers
    if 'EMA9' in df.columns and 'EMA21' in df.columns:
        ema9_prev = safe_get(df, 'EMA9', -2)
        ema21_prev = safe_get(df, 'EMA21', -2)
        ema9_curr = safe_get(df, 'EMA9', -1)
        ema21_curr = safe_get(df, 'EMA21', -1)
        
        if (ema9_prev is not None and ema21_prev is not None and 
            ema9_curr is not None and ema21_curr is not None):
            if safe_compare(ema9_prev, ema21_prev, 'less') and safe_compare(ema9_curr, ema21_curr, 'greater'):
                ma_signals.append("➕ **EMA9 crossed above EMA21**: Bullish signal for short-term momentum.")
                ma_score += 1
            elif safe_compare(ema9_prev, ema21_prev, 'greater') and safe_compare(ema9_curr, ema21_curr, 'less'):
                ma_signals.append("➖ **EMA9 crossed below EMA21**: Bearish signal for short-term momentum.")
                ma_score -= 1
    
    signals['moving_averages'] = 'bullish' if ma_score > 0 else 'bearish' if ma_score < 0 else 'neutral'
    signals['moving_averages_score'] = ma_score
    signals['moving_averages_explanation'] = "\n".join(ma_signals) if ma_signals else "No clear moving average signals."
    
    # RSI signals
    rsi_signals = []
    rsi_score = 0
    
    if 'RSI' in df.columns:
        current_rsi = safe_get(df, 'RSI', -1)
        previous_rsi = safe_get(df, 'RSI', -2)
        
        if current_rsi is not None and previous_rsi is not None:
            if current_rsi < 30:
                rsi_signals.append(f"➕ **RSI is oversold ({current_rsi:.2f})**: Potential buying opportunity.")
                rsi_score += 1
            elif current_rsi > 70:
                rsi_signals.append(f"➖ **RSI is overbought ({current_rsi:.2f})**: Potential selling opportunity.")
                rsi_score -= 1
            
            # RSI trend
            if previous_rsi < 50 and current_rsi > 50:
                rsi_signals.append("➕ **RSI crossed above 50**: Bullish momentum building.")
                rsi_score += 0.5
            elif previous_rsi > 50 and current_rsi < 50:
                rsi_signals.append("➖ **RSI crossed below 50**: Bearish momentum building.")
                rsi_score -= 0.5
        
        # RSI divergence
        if len(df) >= 5:  # Need enough data points to check for divergence
            rsi_5_back = safe_get(df, 'RSI', -5)
            close_curr = safe_get(df, 'Close', -1)
            close_5_back = safe_get(df, 'Close', -5)
            
            if (rsi_5_back is not None and current_rsi is not None and 
                close_curr is not None and close_5_back is not None):
                
                if safe_compare(close_curr, close_5_back, 'greater') and safe_compare(current_rsi, rsi_5_back, 'less'):
                    rsi_signals.append("➖ **Bearish RSI divergence**: Price making higher highs while RSI making lower highs.")
                    rsi_score -= 1
                elif safe_compare(close_curr, close_5_back, 'less') and safe_compare(current_rsi, rsi_5_back, 'greater'):
                    rsi_signals.append("➕ **Bullish RSI divergence**: Price making lower lows while RSI making higher lows.")
                    rsi_score += 1
    
    signals['rsi'] = 'bullish' if rsi_score > 0 else 'bearish' if rsi_score < 0 else 'neutral'
    signals['rsi_score'] = rsi_score
    signals['rsi_explanation'] = "\n".join(rsi_signals) if rsi_signals else "RSI is in neutral territory."
    
    # MACD signals
    macd_signals = []
    macd_score = 0
    
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        macd_line = safe_get(df, 'MACD_12_26_9', -1)
        signal_line = safe_get(df, 'MACDs_12_26_9', -1)
        prev_macd_line = safe_get(df, 'MACD_12_26_9', -2)
        prev_signal_line = safe_get(df, 'MACDs_12_26_9', -2)
        
        # MACD crossover - only process if all values are valid
        if (macd_line is not None and signal_line is not None and 
            prev_macd_line is not None and prev_signal_line is not None):
            
            if safe_compare(prev_macd_line, prev_signal_line, 'less') and safe_compare(macd_line, signal_line, 'greater'):
                macd_signals.append("➕ **MACD crossed above signal line**: Bullish signal.")
                macd_score += 1
            elif safe_compare(prev_macd_line, prev_signal_line, 'greater') and safe_compare(macd_line, signal_line, 'less'):
                macd_signals.append("➖ **MACD crossed below signal line**: Bearish signal.")
                macd_score -= 1
            
            # MACD zero line
            if safe_compare(prev_macd_line, 0, 'less') and safe_compare(macd_line, 0, 'greater'):
                macd_signals.append("➕ **MACD crossed above zero line**: Bullish trend confirmation.")
                macd_score += 0.5
            elif safe_compare(prev_macd_line, 0, 'greater') and safe_compare(macd_line, 0, 'less'):
                macd_signals.append("➖ **MACD crossed below zero line**: Bearish trend confirmation.")
                macd_score -= 0.5
        
        # MACD histogram
        if 'MACDh_12_26_9' in df.columns:
            hist = safe_get(df, 'MACDh_12_26_9', -1)
            prev_hist = safe_get(df, 'MACDh_12_26_9', -2)
            
            if hist is not None and prev_hist is not None:
                if safe_compare(hist, 0, 'greater') and safe_compare(prev_hist, 0, 'less'):
                    macd_signals.append("➕ **MACD histogram turned positive**: Bullish momentum building.")
                    macd_score += 0.5
                elif safe_compare(hist, 0, 'less') and safe_compare(prev_hist, 0, 'greater'):
                    macd_signals.append("➖ **MACD histogram turned negative**: Bearish momentum building.")
                    macd_score -= 0.5
                elif safe_compare(hist, 0, 'greater') and safe_compare(hist, prev_hist, 'greater'):
                    macd_signals.append("➕ **MACD histogram increasing**: Bullish momentum strengthening.")
                    macd_score += 0.25
                elif safe_compare(hist, 0, 'less') and safe_compare(hist, prev_hist, 'less'):
                    macd_signals.append("➖ **MACD histogram decreasing**: Bearish momentum strengthening.")
                    macd_score -= 0.25
    
    signals['macd'] = 'bullish' if macd_score > 0 else 'bearish' if macd_score < 0 else 'neutral'
    signals['macd_score'] = macd_score
    signals['macd_explanation'] = "\n".join(macd_signals) if macd_signals else "No clear MACD signals."
    
    # Bollinger Bands signals
    bb_signals = []
    bb_score = 0
    
    if 'BBL_20_2.0' in df.columns and 'BBM_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
        lower_band = safe_get(df, 'BBL_20_2.0', -1)
        middle_band = safe_get(df, 'BBM_20_2.0', -1)
        upper_band = safe_get(df, 'BBU_20_2.0', -1)
        close_price = safe_get(df, 'Close', -1)
        
        # Only proceed if all values are available
        if (lower_band is not None and middle_band is not None and 
            upper_band is not None and close_price is not None):
            
            # Price near bands
            if safe_compare(close_price, lower_band, 'less') or close_price == lower_band:
                bb_signals.append("➕ **Price at/below lower Bollinger Band**: Potential oversold condition, watch for reversal.")
                bb_score += 0.75
            elif safe_compare(close_price, upper_band, 'greater') or close_price == upper_band:
                bb_signals.append("➖ **Price at/above upper Bollinger Band**: Potential overbought condition, watch for reversal.")
                bb_score -= 0.75
            
            # Band width - volatility
            if 'BBB_20_2.0' in df.columns and len(df) > 5:
                band_width = safe_get(df, 'BBB_20_2.0', -1)
                prev_band_width = safe_get(df, 'BBB_20_2.0', -5)
                
                if (band_width is not None and prev_band_width is not None):
                    if safe_compare(band_width, prev_band_width * 0.8, 'less'):
                        bb_signals.append("📊 **Bollinger Band squeeze**: Decreasing volatility, potential for breakout.")
                    elif safe_compare(band_width, prev_band_width * 1.2, 'greater'):
                        bb_signals.append("📊 **Bollinger Bands widening**: Increasing volatility, trend likely to continue.")
            
            # Bollinger Band bounce
            if len(df) > 2:
                prev_close = safe_get(df, 'Close', -2)
                prev_lower = safe_get(df, 'BBL_20_2.0', -2)
                prev_upper = safe_get(df, 'BBU_20_2.0', -2)
                
                if (prev_close is not None and prev_lower is not None and 
                    prev_upper is not None):
                    
                    # Check for bounce from lower band
                    if ((safe_compare(prev_close, prev_lower, 'less') or prev_close == prev_lower) and 
                        safe_compare(close_price, prev_close, 'greater')):
                        bb_signals.append("➕ **Bounce from lower Bollinger Band**: Potential bullish reversal.")
                        bb_score += 1
                    # Check for bounce from upper band
                    elif ((safe_compare(prev_close, prev_upper, 'greater') or prev_close == prev_upper) and 
                          safe_compare(close_price, prev_close, 'less')):
                        bb_signals.append("➖ **Bounce from upper Bollinger Band**: Potential bearish reversal.")
                        bb_score -= 1
    
    signals['bollinger'] = 'bullish' if bb_score > 0 else 'bearish' if bb_score < 0 else 'neutral'
    signals['bollinger_score'] = bb_score
    signals['bollinger_explanation'] = "\n".join(bb_signals) if bb_signals else "Price is within normal Bollinger Bands range."
    
    # Stochastic signals
    stoch_signals = []
    stoch_score = 0
    
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns:
        k_line = safe_get(df, 'STOCHk_14_3_3', -1)
        d_line = safe_get(df, 'STOCHd_14_3_3', -1)
        prev_k_line = safe_get(df, 'STOCHk_14_3_3', -2)
        prev_d_line = safe_get(df, 'STOCHd_14_3_3', -2)
        
        if k_line is not None and d_line is not None:
            # Overbought/oversold
            if k_line < 20 and d_line < 20:
                stoch_signals.append(f"➕ **Stochastic in oversold territory (K: {k_line:.2f}, D: {d_line:.2f})**: Potential buying opportunity.")
                stoch_score += 0.75
            elif k_line > 80 and d_line > 80:
                stoch_signals.append(f"➖ **Stochastic in overbought territory (K: {k_line:.2f}, D: {d_line:.2f})**: Potential selling opportunity.")
                stoch_score -= 0.75
        
        if (k_line is not None and d_line is not None and 
            prev_k_line is not None and prev_d_line is not None):
            # Stochastic crossover
            if safe_compare(prev_k_line, prev_d_line, 'less') and safe_compare(k_line, d_line, 'greater'):
                stoch_signals.append("➕ **Stochastic %K crossed above %D**: Bullish signal.")
                stoch_score += 1
            elif safe_compare(prev_k_line, prev_d_line, 'greater') and safe_compare(k_line, d_line, 'less'):
                stoch_signals.append("➖ **Stochastic %K crossed below %D**: Bearish signal.")
                stoch_score -= 1
        
        # Divergence
        if len(df) >= 5:
            close_curr = safe_get(df, 'Close', -1) 
            close_5_back = safe_get(df, 'Close', -5)
            k_5_back = safe_get(df, 'STOCHk_14_3_3', -5)
            
            if (close_curr is not None and close_5_back is not None and 
                k_line is not None and k_5_back is not None):
                
                if safe_compare(close_curr, close_5_back, 'greater') and safe_compare(k_line, k_5_back, 'less'):
                    stoch_signals.append("➖ **Bearish Stochastic divergence**: Price making higher highs while Stochastic making lower highs.")
                    stoch_score -= 0.5
                elif safe_compare(close_curr, close_5_back, 'less') and safe_compare(k_line, k_5_back, 'greater'):
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
            current_obv = safe_get(df, 'OBV', -1)
            prev_obv = safe_get(df, 'OBV', -5)
            current_close = safe_get(df, 'Close', -1)
            prev_close = safe_get(df, 'Close', -5)
            
            if (current_obv is not None and prev_obv is not None and 
                current_close is not None and prev_close is not None):
                
                if safe_compare(current_obv, prev_obv, 'greater') and safe_compare(current_close, prev_close, 'greater'):
                    volume_signals.append("➕ **Rising OBV with rising price**: Confirmation of uptrend.")
                    volume_score += 0.5
                elif safe_compare(current_obv, prev_obv, 'less') and safe_compare(current_close, prev_close, 'less'):
                    volume_signals.append("➖ **Falling OBV with falling price**: Confirmation of downtrend.")
                    volume_score -= 0.5
                elif safe_compare(current_obv, prev_obv, 'greater') and safe_compare(current_close, prev_close, 'less'):
                    volume_signals.append("➕ **Rising OBV with falling price**: Potential bullish divergence.")
                    volume_score += 0.75
                elif safe_compare(current_obv, prev_obv, 'less') and safe_compare(current_close, prev_close, 'greater'):
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
