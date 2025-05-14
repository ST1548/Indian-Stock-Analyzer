import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from stock_data_handler import get_stock_data, get_company_info
from fundamental_analysis import perform_fundamental_analysis
from technical_analysis import calculate_technical_indicators, generate_technical_signals
from ml_models import predict_stock_movement
from visualization import (
    plot_stock_price_chart,
    plot_technical_indicators,
    create_signal_dashboard
)
from utils import timeframe_to_period, validate_ticker

# Page configuration
st.set_page_config(
    page_title="Indian Stock Market Analyzer",
    page_icon="📈",
    layout="wide"
)

# App title and description
st.title("Indian Stock Market Analyzer")
st.markdown("""
This application analyzes Indian stocks from NSE and BSE using fundamental and technical analysis to provide 
buy/sell recommendations for intraday and swing trading. Select your stock symbol in the sidebar and
choose between NSE and BSE exchanges.
""")

# Sidebar for user inputs
with st.sidebar:
    st.header("Stock Selection")
    st.markdown("""
    Enter stock symbol for Indian markets:
    - For NSE stocks: Use symbol directly or with .NS suffix (e.g. RELIANCE or RELIANCE.NS)
    - For BSE stocks: Use symbol with .BO suffix (e.g. RELIANCE.BO)
    """)
    
    # Common Indian stocks for quick selection
    st.subheader("Popular Indian Stocks")
    st.markdown("""
    **NSE Nifty50 Stocks:**
    - RELIANCE - Reliance Industries
    - TCS - Tata Consultancy Services
    - HDFCBANK - HDFC Bank
    - INFY - Infosys
    - ICICIBANK - ICICI Bank
    - HINDUNILVR - Hindustan Unilever
    - BAJFINANCE - Bajaj Finance
    - SBIN - State Bank of India
    - BHARTIARTL - Bharti Airtel
    - ADANIENT - Adani Enterprises
    """)
    
    ticker_input = st.text_input("Enter Stock Symbol", "RELIANCE").upper()
    
    st.header("Exchange")
    exchange = st.radio(
        "Select Exchange",
        ["NSE", "BSE"],
        index=0
    )
    
    if '.' not in ticker_input:
        if exchange == "NSE":
            ticker_suffix = ".NS"
        else:  # BSE
            ticker_suffix = ".BO"
        
        full_ticker = f"{ticker_input}{ticker_suffix}"
        st.caption(f"Your selected ticker: **{full_ticker}**")
    else:
        full_ticker = ticker_input
        st.caption(f"Using provided ticker: **{full_ticker}**")
    
    st.header("Analysis Parameters")
    timeframe = st.selectbox(
        "Select Timeframe",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y"],
        index=2
    )
    
    interval = st.selectbox(
        "Select Interval",
        ["1m", "5m", "15m", "30m", "60m", "1d"],
        index=5
    )
    
    # Only show intraday option for intervals less than 1d
    show_intraday = interval != "1d"
    
    st.header("Trading Strategy")
    if show_intraday:
        analyze_intraday = st.checkbox("Analyze for Intraday Trading", True)
    else:
        analyze_intraday = False
        
    analyze_swing = st.checkbox("Analyze for Swing Trading", True)
    
    st.button("Run Analysis", key="run_analysis")

# Main content
try:
    # Validate ticker
    if not ticker_input:
        st.warning("Please enter a stock symbol.")
        st.stop()
    
    # Use the full ticker with exchange suffix if it doesn't have one already
    if '.' not in ticker_input:
        if exchange == "NSE":
            ticker_suffix = ".NS"
        else:  # BSE
            ticker_suffix = ".BO"
        full_ticker = f"{ticker_input}{ticker_suffix}"
    else:
        full_ticker = ticker_input
        
    ticker_valid = validate_ticker(full_ticker)
    if not ticker_valid:
        st.error(f"Invalid ticker symbol: {full_ticker}. Please enter a valid symbol.")
        st.stop()
    
    with st.spinner(f"Fetching data for {full_ticker}..."):
        # Get stock data
        period = timeframe_to_period(timeframe)
        
        try:
            stock_data = get_stock_data(full_ticker, period, interval)
            
            if stock_data.empty:
                st.error(f"No data available for {full_ticker}. Please try another stock symbol.")
                st.stop()
                
            # Verify that the stock data has required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in stock_data.columns]
            
            if missing_columns:
                st.error(f"Missing required data columns: {', '.join(missing_columns)}. Please try another stock or time period.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error processing data for {full_ticker}: {str(e)}")
            st.error("Please try another stock symbol or check your internet connection.")
            st.stop()
        
        # Get company information
        company_info = get_company_info(full_ticker)
    
    # Display company information and current price
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader(f"{company_info.get('longName', full_ticker)}")
        st.markdown(f"**Sector:** {company_info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {company_info.get('industry', 'N/A')}")
        st.markdown(f"**Exchange:** {company_info.get('exchange', 'N/A')}")
        st.markdown(f"**Currency:** {company_info.get('currency', 'INR')}")
        
    with col2:
        try:
            current_price = stock_data['Close'].iloc[-1]
            previous_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0
            
            price_color = "green" if price_change >= 0 else "red"
            change_icon = "▲" if price_change >= 0 else "▼"
            
            # Get currency symbol (default to ₹ for Indian stocks)
            currency_symbol = "₹"
            if company_info.get('currency') == 'USD':
                currency_symbol = "$"
            
            st.metric(
                label="Current Price",
                value=f"{currency_symbol}{current_price:.2f}",
                delta=f"{change_icon} {currency_symbol}{abs(price_change):.2f} ({price_change_pct:.2f}%)"
            )
        except Exception as e:
            st.error(f"Error displaying price information: {str(e)}")
            st.metric(
                label="Current Price",
                value="N/A",
                delta="N/A"
            )
        
    with col3:
        try:
            # Get currency symbol (default to ₹ for Indian stocks)
            currency_symbol = "₹"
            if company_info.get('currency') == 'USD':
                currency_symbol = "$"
                
            # Format market cap
            market_cap = company_info.get('marketCap', 0)
            if market_cap > 0:
                if market_cap > 1e9:
                    market_cap_str = f"{currency_symbol}{market_cap / 1e9:.2f}B"
                else:
                    market_cap_str = f"{currency_symbol}{market_cap / 1e6:.2f}M"
            else:
                market_cap_str = "N/A"
            
            # Format volume with error handling
            try:
                volume = stock_data['Volume'].iloc[-1]
                volume_str = f"{volume:,.0f}"
            except:
                volume_str = "N/A"
                
            st.markdown(f"**Market Cap:** {market_cap_str}")
            st.markdown(f"**Volume:** {volume_str}")
            
        except Exception as e:
            st.markdown("**Market Cap:** N/A")
            st.markdown("**Volume:** N/A")
        st.markdown(f"**52w Range:** ${company_info.get('fiftyTwoWeekLow', 0):.2f} - ${company_info.get('fiftyTwoWeekHigh', 0):.2f}")
    
    # Stock price chart
    st.subheader("Stock Price History")
    fig = plot_stock_price_chart(stock_data, full_ticker, timeframe)
    st.plotly_chart(fig, use_container_width=True)
    
    # Fundamental Analysis
    st.header("Fundamental Analysis")
    fundamental_data = perform_fundamental_analysis(full_ticker)
    
    # Display fundamental analysis in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Valuation Metrics")
        metrics = {
            "P/E Ratio": fundamental_data.get("trailingPE", "N/A"),
            "Forward P/E": fundamental_data.get("forwardPE", "N/A"),
            "PEG Ratio": fundamental_data.get("pegRatio", "N/A"),
            "Price to Book": fundamental_data.get("priceToBook", "N/A"),
            "Price to Sales": fundamental_data.get("priceToSalesTrailing12Months", "N/A")
        }
        
        for key, value in metrics.items():
            # Handle None values and convert to "N/A"
            if value is None:
                value = "N/A"
            # Only try to format if it's not "N/A" and not None
            elif value != "N/A":
                try:
                    value = f"{float(value):.2f}"
                except (TypeError, ValueError):
                    # If conversion fails, display as is
                    value = "N/A"
            st.markdown(f"**{key}:** {value}")
    
    with col2:
        st.subheader("Profitability Metrics")
        metrics = {
            "Profit Margin": fundamental_data.get("profitMargins", "N/A"),
            "Operating Margin": fundamental_data.get("operatingMargins", "N/A"),
            "ROE": fundamental_data.get("returnOnEquity", "N/A"),
            "ROA": fundamental_data.get("returnOnAssets", "N/A"),
            "EPS Growth": fundamental_data.get("earningsGrowth", "N/A")
        }
        
        for key, value in metrics.items():
            # Handle None values and convert to "N/A"
            if value is None:
                value = "N/A"
            # Only try to format if it's not "N/A" and not None
            elif value != "N/A":
                try:
                    # All values need to be shown as percentages
                    value = f"{float(value) * 100:.2f}%"
                except (TypeError, ValueError):
                    # If conversion fails, display as is
                    value = "N/A"
            st.markdown(f"**{key}:** {value}")
    
    with col3:
        st.subheader("Financial Health")
        metrics = {
            "Debt to Equity": fundamental_data.get("debtToEquity", "N/A"),
            "Current Ratio": fundamental_data.get("currentRatio", "N/A"),
            "Quick Ratio": fundamental_data.get("quickRatio", "N/A"),
            "Dividend Yield": fundamental_data.get("dividendYield", "N/A"),
            "Payout Ratio": fundamental_data.get("payoutRatio", "N/A")
        }
        
        for key, value in metrics.items():
            # Handle None values and convert to "N/A"
            if value is None:
                value = "N/A"
            # Only try to format if it's not "N/A"
            elif value != "N/A":
                try:
                    if key in ["Dividend Yield", "Payout Ratio"]:
                        # Format as percentage
                        value = f"{float(value) * 100:.2f}%"
                    else:
                        # Format as decimal
                        value = f"{float(value):.2f}"
                except (TypeError, ValueError):
                    # If conversion fails, display as is
                    value = "N/A"
            st.markdown(f"**{key}:** {value}")
    
    # Fundamental Analysis AI Interpretation
    st.subheader("Fundamental Analysis Interpretation")
    
    # Create fundamental scores based on common financial metrics
    pe_score = 0
    try:
        pe = fundamental_data.get("trailingPE")
        if pe is not None:
            pe_float = float(pe)
            if pe_float < 15:
                pe_score = 1  # Potentially undervalued
            elif pe_float > 30:
                pe_score = -1  # Potentially overvalued
    except (TypeError, ValueError):
        # If conversion fails, leave the score as 0
        pass
    
    growth_score = 0
    try:
        growth = fundamental_data.get("earningsGrowth")
        if growth is not None:
            growth_float = float(growth)
            if growth_float > 0.1:  # 10% growth
                growth_score = 1
            elif growth_float < 0:
                growth_score = -1
    except (TypeError, ValueError):
        # If conversion fails, leave the score as 0
        pass
    
    profitability_score = 0
    try:
        margin = fundamental_data.get("profitMargins")
        if margin is not None:
            margin_float = float(margin)
            if margin_float > 0.15:  # 15% profit margin
                profitability_score = 1
            elif margin_float < 0.05:
                profitability_score = -1
    except (TypeError, ValueError):
        # If conversion fails, leave the score as 0
        pass
    
    debt_score = 0
    try:
        debt = fundamental_data.get("debtToEquity")
        if debt is not None:
            debt_float = float(debt)
            if debt_float < 50:  # Low debt
                debt_score = 1
            elif debt_float > 100:  # High debt
                debt_score = -1
    except (TypeError, ValueError):
        # If conversion fails, leave the score as 0
        pass
    
    # Aggregate fundamental score
    fundamental_score = pe_score + growth_score + profitability_score + debt_score
    
    if fundamental_score > 2:
        st.markdown("📈 **Strong fundamentals**: The company shows excellent financial health with good valuation metrics, strong growth, and low debt.")
    elif fundamental_score > 0:
        st.markdown("✅ **Solid fundamentals**: The company appears to be in good financial health with reasonable valuation.")
    elif fundamental_score == 0:
        st.markdown("⚖️ **Neutral fundamentals**: The company shows a mix of positive and negative financial indicators.")
    elif fundamental_score > -2:
        st.markdown("⚠️ **Caution on fundamentals**: Some financial metrics raise concerns about valuation or financial health.")
    else:
        st.markdown("🔻 **Weak fundamentals**: The company shows multiple concerning financial indicators that may impact long-term performance.")
    
    # Technical Analysis
    st.header("Technical Analysis")
    
    # Calculate technical indicators
    with st.spinner("Calculating technical indicators..."):
        try:
            technical_data = calculate_technical_indicators(stock_data)
            
            # Verify technical data was calculated successfully
            if technical_data is None or technical_data.empty:
                st.error("Failed to calculate technical indicators. Data may be insufficient.")
                st.stop()
                
            technical_signals = generate_technical_signals(technical_data)
            
            # Verify signals were generated successfully
            if not technical_signals:
                st.warning("Unable to generate reliable technical signals. Data may be insufficient.")
                # Continue execution but with a warning
        except Exception as e:
            st.error(f"Error in technical analysis: {str(e)}")
            st.error("Technical indicators could not be calculated. Please try another stock or timeframe.")
            st.stop()
    
    # Plot technical indicators
    st.subheader("Technical Indicators")
    tabs = st.tabs([
        "Moving Averages", "RSI", "MACD", 
        "Bollinger Bands", "Stochastic", "Volume"
    ])
    
    with tabs[0]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'moving_averages'),
            use_container_width=True
        )
        st.markdown(technical_signals['moving_averages_explanation'])
    
    with tabs[1]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'rsi'),
            use_container_width=True
        )
        st.markdown(technical_signals['rsi_explanation'])
    
    with tabs[2]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'macd'),
            use_container_width=True
        )
        st.markdown(technical_signals['macd_explanation'])
    
    with tabs[3]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'bollinger'),
            use_container_width=True
        )
        st.markdown(technical_signals['bollinger_explanation'])
    
    with tabs[4]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'stochastic'),
            use_container_width=True
        )
        st.markdown(technical_signals['stochastic_explanation'])
    
    with tabs[5]:
        st.plotly_chart(
            plot_technical_indicators(technical_data, full_ticker, 'volume'),
            use_container_width=True
        )
        st.markdown(technical_signals['volume_explanation'])
    
    # Signal Dashboard
    st.header("Signal Dashboard")
    signal_dashboard = create_signal_dashboard(technical_signals)
    st.plotly_chart(signal_dashboard, use_container_width=True)
    
    # AI Trading Recommendations
    st.header("AI Trading Recommendations")
    
    # ML model predictions
    with st.spinner("Generating AI recommendations..."):
        intraday_prediction = None
        swing_prediction = None
        
        if analyze_intraday and show_intraday:
            intraday_prediction = predict_stock_movement(
                technical_data, 
                full_ticker, 
                'intraday'
            )
        
        if analyze_swing:
            swing_prediction = predict_stock_movement(
                technical_data, 
                full_ticker, 
                'swing'
            )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if analyze_intraday and show_intraday:
            st.subheader("Intraday Trading")
            
            recommendation = intraday_prediction['recommendation']
            confidence = intraday_prediction['confidence']
            
            if recommendation == 'buy':
                st.markdown(f"### 🟢 **BUY** (Confidence: {confidence:.1%})")
            elif recommendation == 'sell':
                st.markdown(f"### 🔴 **SELL** (Confidence: {confidence:.1%})")
            else:
                st.markdown(f"### ⚪ **HOLD/NEUTRAL** (Confidence: {confidence:.1%})")
            
            st.markdown("#### Key Factors:")
            st.markdown(intraday_prediction['reasoning'])
        elif show_intraday:
            st.subheader("Intraday Trading")
            st.info("Intraday analysis is not selected. Enable it in the sidebar to see recommendations.")
        else:
            st.subheader("Intraday Trading")
            st.info("Intraday analysis is not available for daily interval data. Select a smaller interval for intraday analysis.")
    
    with col2:
        if analyze_swing:
            st.subheader("Swing Trading")
            
            recommendation = swing_prediction['recommendation']
            confidence = swing_prediction['confidence']
            
            if recommendation == 'buy':
                st.markdown(f"### 🟢 **BUY** (Confidence: {confidence:.1%})")
            elif recommendation == 'sell':
                st.markdown(f"### 🔴 **SELL** (Confidence: {confidence:.1%})")
            else:
                st.markdown(f"### ⚪ **HOLD/NEUTRAL** (Confidence: {confidence:.1%})")
            
            st.markdown("#### Key Factors:")
            st.markdown(swing_prediction['reasoning'])
        else:
            st.subheader("Swing Trading")
            st.info("Swing analysis is not selected. Enable it in the sidebar to see recommendations.")
    
    # Disclaimer
    st.markdown("---")
    st.caption("""
    **Disclaimer**: This tool provides analysis of Indian stocks from NSE and BSE for informational purposes only and does not constitute 
    investment advice. All market data is sourced from Yahoo Finance. The analysis and recommendations are algorithmic in nature and may not 
    account for all market factors. Always conduct your own research before making investment decisions in the Indian stock market.
    """)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    import traceback
    st.exception(e)
