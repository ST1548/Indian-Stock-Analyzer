import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from utils import format_indian_ticker

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
        # Format ticker for Indian stock market if needed
        formatted_ticker = format_indian_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        data = stock.history(period=period, interval=interval)
        
        # Handle empty data
        if data.empty:
            # Try alternative exchanges if the symbol was not provided with an exchange suffix
            if '.' not in ticker:
                # Try NSE
                nse_ticker = f"{ticker}.NS"
                stock_nse = yf.Ticker(nse_ticker)
                data_nse = stock_nse.history(period=period, interval=interval)
                
                if not data_nse.empty:
                    data = data_nse
                else:
                    # Try BSE
                    bse_ticker = f"{ticker}.BO"
                    stock_bse = yf.Ticker(bse_ticker)
                    data_bse = stock_bse.history(period=period, interval=interval)
                    
                    if not data_bse.empty:
                        data = data_bse
        
        # If still empty, return empty DataFrame
        if data.empty:
            return pd.DataFrame()
        
        # Handle the index (which could be Datetime or Date)
        try:
            # Reset index to make the date a column
            data = data.reset_index()
            
            # Check if 'Date' or 'Datetime' is in columns
            date_col = None
            if 'Date' in data.columns:
                date_col = 'Date'
            elif 'Datetime' in data.columns:
                date_col = 'Datetime'
            
            # If we found a date column, convert and set as index
            if date_col:
                if not pd.api.types.is_datetime64_any_dtype(data[date_col]):
                    data[date_col] = pd.to_datetime(data[date_col])
                data = data.set_index(date_col)
        except Exception as e:
            print(f"Warning: Issue with date handling: {e}")
            # If there was an error, just return as is (might already have date as index)
            pass
        
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
        # Format ticker for Indian stock market if needed
        formatted_ticker = format_indian_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        info = stock.info
        
        # Check if we got valid info back
        if not info or 'regularMarketPrice' not in info or info.get('regularMarketPrice', 0) == 0:
            # Try alternative exchanges if the symbol was not provided with an exchange suffix
            if '.' not in ticker:
                # Try NSE
                nse_ticker = f"{ticker}.NS"
                stock_nse = yf.Ticker(nse_ticker)
                info_nse = stock_nse.info
                
                if info_nse and 'regularMarketPrice' in info_nse and info_nse.get('regularMarketPrice', 0) > 0:
                    info = info_nse
                    formatted_ticker = nse_ticker
                else:
                    # Try BSE
                    bse_ticker = f"{ticker}.BO"
                    stock_bse = yf.Ticker(bse_ticker)
                    info_bse = stock_bse.info
                    
                    if info_bse and 'regularMarketPrice' in info_bse and info_bse.get('regularMarketPrice', 0) > 0:
                        info = info_bse
                        formatted_ticker = bse_ticker
        
        # Filter relevant information
        relevant_info = {
            'longName': info.get('longName', formatted_ticker),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'marketCap': info.get('marketCap', 'N/A'),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 'N/A'),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            'website': info.get('website', 'N/A'),
            'exchange': info.get('exchange', 'N/A'),
            'currency': info.get('currency', 'INR'),  # Default to INR for Indian stocks
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
            'exchange': 'N/A',
            'currency': 'INR',  # Default to INR for Indian stocks
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
        # Format ticker for Indian stock market if needed
        formatted_ticker = format_indian_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        earnings = stock.earnings
        
        # If earnings data is empty, try alternative exchanges
        if (not isinstance(earnings, pd.DataFrame) or earnings.empty) and '.' not in ticker:
            # Try NSE
            nse_ticker = f"{ticker}.NS"
            stock_nse = yf.Ticker(nse_ticker)
            earnings_nse = stock_nse.earnings
            
            if isinstance(earnings_nse, pd.DataFrame) and not earnings_nse.empty:
                earnings = earnings_nse
            else:
                # Try BSE
                bse_ticker = f"{ticker}.BO"
                stock_bse = yf.Ticker(bse_ticker)
                earnings_bse = stock_bse.earnings
                
                if isinstance(earnings_bse, pd.DataFrame) and not earnings_bse.empty:
                    earnings = earnings_bse
        
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
        # Format ticker for Indian stock market if needed
        formatted_ticker = format_indian_ticker(ticker)
        
        stock = yf.Ticker(formatted_ticker)
        news = stock.news
        
        # If news is empty, try alternative exchanges
        if not news and '.' not in ticker:
            # Try NSE
            nse_ticker = f"{ticker}.NS"
            stock_nse = yf.Ticker(nse_ticker)
            news_nse = stock_nse.news
            
            if news_nse:
                news = news_nse
            else:
                # Try BSE
                bse_ticker = f"{ticker}.BO"
                stock_bse = yf.Ticker(bse_ticker)
                news_bse = stock_bse.news
                
                if news_bse:
                    news = news_bse
        
        if not news:
            return []
        
        # Return limited number of news items
        return news[:limit]
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []
