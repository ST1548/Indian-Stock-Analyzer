import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def timeframe_to_period(timeframe):
    """
    Convert a timeframe string to a period string for yfinance.
    
    Args:
        timeframe (str): Timeframe string (e.g., "1d", "5d", "1mo", etc.)
        
    Returns:
        str: Period string for yfinance
    """
    # Direct mapping for standard periods
    if timeframe in ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]:
        return timeframe
    
    # Custom mapping for non-standard periods
    mapping = {
        "1w": "7d",
        "2w": "14d"
    }
    
    return mapping.get(timeframe, "1mo")  # Default to 1mo if not found

def format_indian_ticker(ticker):
    """
    Format Indian stock tickers to be compatible with Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        str: Formatted ticker symbol
    """
    # If ticker already contains a dot or ends with .NS or .BO, return as is
    if '.' in ticker or ticker.endswith(('.NS', '.BO')):
        return ticker
        
    # For Indian stocks, append .NS (NSE) by default
    return f"{ticker}.NS"

def validate_ticker(ticker):
    """
    Validate if a ticker symbol exists on Yahoo Finance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        # First try as provided
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Check if we got valid info back
        if ('symbol' in info or 'regularMarketPrice' in info) and info.get('regularMarketPrice', 0) > 0:
            return True
            
        # If not valid, try with .NS suffix for Indian stocks
        if '.' not in ticker:
            stock_ns = yf.Ticker(f"{ticker}.NS")
            info_ns = stock_ns.info
            if ('symbol' in info_ns or 'regularMarketPrice' in info_ns) and info_ns.get('regularMarketPrice', 0) > 0:
                return True
                
            # Try with .BO suffix (Bombay Stock Exchange)
            stock_bo = yf.Ticker(f"{ticker}.BO")
            info_bo = stock_bo.info
            if ('symbol' in info_bo or 'regularMarketPrice' in info_bo) and info_bo.get('regularMarketPrice', 0) > 0:
                return True
        
        # If we got a dict back but no valid data, it might not be valid
        return False
    except:
        return False

def format_large_number(num):
    """
    Format large numbers into K, M, B, T format.
    
    Args:
        num (float): Number to format
        
    Returns:
        str: Formatted number string
    """
    if num is None:
        return "N/A"
    
    magnitude = 0
    labels = ["", "K", "M", "B", "T"]
    
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    
    return f"{num:.2f}{labels[magnitude]}"

def get_trading_periods(interval):
    """
    Get market open/close times for a given interval.
    
    Args:
        interval (str): Time interval (e.g., "1d", "1h", "15m")
        
    Returns:
        tuple: (start_time, end_time) in market hours
    """
    # Current date
    now = datetime.now()
    today = now.date()
    
    # Standard market hours (9:30 AM - 4:00 PM Eastern Time)
    market_open = datetime.combine(today, datetime.strptime("9:30", "%H:%M").time())
    market_close = datetime.combine(today, datetime.strptime("16:00", "%H:%M").time())
    
    # For daily intervals, use entire market hours
    if interval in ["1d", "1wk", "1mo"]:
        return market_open, market_close
    
    # For hourly intervals
    if interval in ["1h", "60m"]:
        # If current time is before market close, use current time as end
        if now < market_close:
            end_time = now
        else:
            end_time = market_close
        
        # Start time is 6.5 hours before end (full trading day)
        start_time = end_time - timedelta(hours=6, minutes=30)
        if start_time < market_open:
            start_time = market_open
        
        return start_time, end_time
    
    # For minute intervals
    if interval in ["1m", "5m", "15m", "30m"]:
        # For intraday, focus on recent data
        if interval == "1m":
            lookback_minutes = 60  # Last hour
        elif interval == "5m":
            lookback_minutes = 120  # Last 2 hours
        elif interval == "15m":
            lookback_minutes = 240  # Last 4 hours
        else:  # 30m
            lookback_minutes = 390  # Full trading day (6.5 hours)
        
        # If current time is before market close, use current time as end
        if now < market_close:
            end_time = now
        else:
            end_time = market_close
        
        # Start time is lookback_minutes before end
        start_time = end_time - timedelta(minutes=lookback_minutes)
        if start_time < market_open:
            start_time = market_open
        
        return start_time, end_time
    
    # Default to full market hours
    return market_open, market_close

def calculate_returns(prices, periods=[1, 5, 21, 63, 252]):
    """
    Calculate returns over different periods.
    
    Args:
        prices (pd.Series): Series of price data
        periods (list): List of periods to calculate returns for
        
    Returns:
        dict: Dictionary of returns for each period
    """
    returns = {}
    
    for period in periods:
        if len(prices) > period:
            ret = (prices.iloc[-1] / prices.iloc[-period-1] - 1) * 100
            returns[f"{period}d"] = ret
    
    return returns
