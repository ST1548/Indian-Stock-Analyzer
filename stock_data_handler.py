import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def get_stock_data(ticker, period='1mo', interval='1d'):
    """
    Fetches historical stock data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Period of historical data to fetch
        interval (str): Interval between data points
        
    Returns:
        pd.DataFrame: DataFrame containing the stock data
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        # Handle empty data
        if data.empty:
            return pd.DataFrame()
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        # Convert Date to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            data['Date'] = pd.to_datetime(data['Date'])
            
        # Set Date as index again
        data = data.set_index('Date')
        
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def get_company_info(ticker):
    """
    Fetches company information using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing the company information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Filter relevant information
        relevant_info = {
            'longName': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'website': info.get('website', 'N/A'),
        }
        
        return relevant_info
    except Exception as e:
        print(f"Error fetching company info for {ticker}: {e}")
        return {
            'longName': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'marketCap': 'N/A',
            'fiftyTwoWeekLow': 'N/A',
            'fiftyTwoWeekHigh': 'N/A',
            'beta': 'N/A',
            'website': 'N/A',
        }

def get_earnings_data(ticker):
    """
    Fetches earnings data using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        pd.DataFrame: DataFrame containing earnings data
    """
    try:
        stock = yf.Ticker(ticker)
        earnings = stock.earnings
        
        if isinstance(earnings, pd.DataFrame) and not earnings.empty:
            return earnings
        return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching earnings data for {ticker}: {e}")
        return pd.DataFrame()

def get_news(ticker, limit=5):
    """
    Fetches recent news using yfinance.
    
    Args:
        ticker (str): Stock ticker symbol
        limit (int): Maximum number of news items to return
        
    Returns:
        list: List of news items
    """
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        if not news:
            return []
        
        # Return limited number of news items
        return news[:limit]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []
