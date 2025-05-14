import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime

def plot_stock_price_chart(stock_data, ticker, timeframe):
    """
    Creates a candlestick chart of stock prices.
    
    Args:
        stock_data (pd.DataFrame): DataFrame with stock price data
        ticker (str): Stock ticker symbol
        timeframe (str): Timeframe of the data
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.02, 
                         row_heights=[0.7, 0.3],
                         subplot_titles=(f"{ticker} - {timeframe} Chart", "Volume"))
    
    # Add candlestick trace
    fig.add_trace(
        go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add volume trace
    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in stock_data.iterrows()]
    
    fig.add_trace(
        go.Bar(
            x=stock_data.index,
            y=stock_data['Volume'],
            marker_color=colors,
            name="Volume"
        ),
        row=2, col=1
    )
    
    # Add moving averages if available
    if 'SMA20' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA20'],
                line=dict(color='blue', width=1),
                name="SMA20"
            ),
            row=1, col=1
        )
    
    if 'SMA50' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA50'],
                line=dict(color='orange', width=1),
                name="SMA50"
            ),
            row=1, col=1
        )
    
    if 'SMA200' in stock_data.columns:
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['SMA200'],
                line=dict(color='purple', width=1),
                name="SMA200"
            ),
            row=1, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Price Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def plot_technical_indicators(data, ticker, indicator_type):
    """
    Creates plots for various technical indicators.
    
    Args:
        data (pd.DataFrame): DataFrame with technical indicators
        ticker (str): Stock ticker symbol
        indicator_type (str): Type of indicator to plot
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if indicator_type == 'moving_averages':
        # Plot price with multiple moving averages
        fig = go.Figure()
        
        # Add price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add moving averages
        if 'SMA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    mode='lines',
                    name='SMA20',
                    line=dict(color='blue', width=1)
                )
            )
        
        if 'SMA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    mode='lines',
                    name='SMA50',
                    line=dict(color='orange', width=1)
                )
            )
        
        if 'SMA200' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA200'],
                    mode='lines',
                    name='SMA200',
                    line=dict(color='purple', width=1)
                )
            )
        
        if 'EMA9' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA9'],
                    mode='lines',
                    name='EMA9',
                    line=dict(color='green', width=1, dash='dash')
                )
            )
        
        if 'EMA21' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['EMA21'],
                    mode='lines',
                    name='EMA21',
                    line=dict(color='red', width=1, dash='dash')
                )
            )
        
        fig.update_layout(
            title=f"{ticker} - Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    elif indicator_type == 'rsi':
        # Plot RSI
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{ticker} - Price", "RSI"))
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add RSI trace
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    mode='lines',
                    name='RSI (14)',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[70, 70],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name='Overbought (70)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[30, 30],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    name='Oversold (30)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[50, 50],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name='Centerline (50)'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis ranges for RSI
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        
        return fig
    
    elif indicator_type == 'macd':
        # Plot MACD
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{ticker} - Price", "MACD"))
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add MACD traces
        if 'MACD_12_26_9' in data.columns and 'MACDs_12_26_9' in data.columns:
            # MACD Line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACD_12_26_9'],
                    mode='lines',
                    name='MACD Line',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # Signal Line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['MACDs_12_26_9'],
                    mode='lines',
                    name='Signal Line',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Histogram
            if 'MACDh_12_26_9' in data.columns:
                colors = ['green' if val >= 0 else 'red' for val in data['MACDh_12_26_9']]
                
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=data['MACDh_12_26_9'],
                        name='Histogram',
                        marker_color=colors
                    ),
                    row=2, col=1
                )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        return fig
    
    elif indicator_type == 'bollinger':
        # Plot Bollinger Bands
        fig = go.Figure()
        
        # Add price
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            )
        )
        
        # Add Bollinger Bands
        if 'BBU_20_2.0' in data.columns and 'BBM_20_2.0' in data.columns and 'BBL_20_2.0' in data.columns:
            # Upper band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BBU_20_2.0'],
                    mode='lines',
                    name='Upper Band',
                    line=dict(color='red', width=1)
                )
            )
            
            # Middle band (SMA20)
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BBM_20_2.0'],
                    mode='lines',
                    name='Middle Band (SMA20)',
                    line=dict(color='blue', width=1)
                )
            )
            
            # Lower band
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BBL_20_2.0'],
                    mode='lines',
                    name='Lower Band',
                    line=dict(color='green', width=1)
                )
            )
            
            # Add band width if available
            if 'BBB_20_2.0' in data.columns:
                # Normalize band width for visualization
                bb_width_norm = data['BBB_20_2.0'] / data['BBB_20_2.0'].max() * (data['Close'].max() * 0.1)
                bb_width_base = data['Close'].min() - (data['Close'].max() * 0.15)
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=bb_width_base + bb_width_norm,
                        mode='lines',
                        name='Band Width (scaled)',
                        line=dict(color='purple', width=1),
                        visible='legendonly'  # Hide by default
                    )
                )
        
        fig.update_layout(
            title=f"{ticker} - Bollinger Bands",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    elif indicator_type == 'stochastic':
        # Plot Stochastic Oscillator
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{ticker} - Price", "Stochastic Oscillator"))
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add Stochastic traces
        if 'STOCHk_14_3_3' in data.columns and 'STOCHd_14_3_3' in data.columns:
            # %K Line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['STOCHk_14_3_3'],
                    mode='lines',
                    name='%K Line',
                    line=dict(color='blue', width=1)
                ),
                row=2, col=1
            )
            
            # %D Line
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['STOCHd_14_3_3'],
                    mode='lines',
                    name='%D Line',
                    line=dict(color='red', width=1)
                ),
                row=2, col=1
            )
            
            # Add overbought/oversold lines
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[80, 80],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dash'),
                    name='Overbought (80)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[20, 20],
                    mode='lines',
                    line=dict(color='green', width=1, dash='dash'),
                    name='Oversold (20)'
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[50, 50],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name='Centerline (50)'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axis ranges for Stochastic
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Stochastic", range=[0, 100], row=2, col=1)
        
        return fig
    
    elif indicator_type == 'volume':
        # Plot Volume Indicators
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, 
                           row_heights=[0.7, 0.3],
                           subplot_titles=(f"{ticker} - Price and Volume", "OBV"))
        
        # Add price trace
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='black', width=2)
            ),
            row=1, col=1
        )
        
        # Add volume bars on the same pane
        colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.3,
                yaxis="y2"
            ),
            row=1, col=1
        )
        
        # Add volume SMA
        if 'Volume_SMA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volume_SMA20'],
                    mode='lines',
                    name='Volume SMA20',
                    line=dict(color='blue', width=1),
                    yaxis="y2"
                ),
                row=1, col=1
            )
        
        # Add OBV
        if 'OBV' in data.columns:
            # Normalize OBV for visualization
            obv_min = data['OBV'].min()
            obv_max = data['OBV'].max()
            normalized_obv = (data['OBV'] - obv_min) / (obv_max - obv_min) * 100 if obv_max > obv_min else data['OBV']
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_obv,
                    mode='lines',
                    name='OBV (normalized)',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            template="plotly_white",
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Create a secondary y-axis for volume
        fig.update_layout(
            yaxis=dict(
                title="Price ($)",
                domain=[0, 0.7]
            ),
            yaxis2=dict(
                title="Volume",
                domain=[0, 0.7],
                anchor="x",
                overlaying="y",
                side="right"
            )
        )
        
        fig.update_yaxes(title_text="Normalized OBV", row=2, col=1)
        
        return fig
    
    else:
        # Default empty figure if indicator not supported
        fig = go.Figure()
        fig.update_layout(
            title=f"Indicator '{indicator_type}' not supported",
            template="plotly_white",
            height=500
        )
        return fig

def create_signal_dashboard(signals):
    """
    Creates a dashboard visualization for technical signals.
    
    Args:
        signals (dict): Dictionary containing technical signals
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    # Extract signal types and scores
    signal_types = [
        'Moving Averages', 'RSI', 'MACD',
        'Bollinger Bands', 'Stochastic', 'Volume'
    ]
    
    scores = [
        signals.get('moving_averages_score', 0),
        signals.get('rsi_score', 0),
        signals.get('macd_score', 0),
        signals.get('bollinger_score', 0),
        signals.get('stochastic_score', 0),
        signals.get('volume_score', 0)
    ]
    
    # Map scores to signal strength (1 to 5)
    normalized_scores = []
    colors = []
    
    for score in scores:
        if score >= 2:
            norm_score = 5  # Strong Buy
            color = 'green'
        elif score > 0:
            norm_score = 4  # Buy
            color = 'lightgreen'
        elif score == 0:
            norm_score = 3  # Neutral
            color = 'gray'
        elif score > -2:
            norm_score = 2  # Sell
            color = 'lightcoral'
        else:
            norm_score = 1  # Strong Sell
            color = 'red'
        
        normalized_scores.append(norm_score)
        colors.append(color)
    
    # Calculate overall signal
    overall_score = signals.get('overall_score', 0)
    if overall_score >= 2:
        overall_normalized = 5  # Strong Buy
        overall_color = 'green'
        overall_text = 'STRONG BUY'
    elif overall_score > 0:
        overall_normalized = 4  # Buy
        overall_color = 'lightgreen'
        overall_text = 'BUY'
    elif overall_score == 0:
        overall_normalized = 3  # Neutral
        overall_color = 'gray'
        overall_text = 'NEUTRAL'
    elif overall_score > -2:
        overall_normalized = 2  # Sell
        overall_color = 'lightcoral'
        overall_text = 'SELL'
    else:
        overall_normalized = 1  # Strong Sell
        overall_color = 'red'
        overall_text = 'STRONG SELL'
    
    # Create the figure
    fig = go.Figure()
    
    # Add individual indicator bars
    fig.add_trace(go.Bar(
        x=signal_types,
        y=normalized_scores,
        marker_color=colors,
        text=['Strong Buy' if s==5 else 'Buy' if s==4 else 'Neutral' if s==3 else 'Sell' if s==2 else 'Strong Sell' for s in normalized_scores],
        textposition='auto',
        name='Indicator Signals'
    ))
    
    # Add overall recommendation
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=overall_normalized,
        domain={'x': [0.5, 1], 'y': [0, 0.3]},
        title={'text': "Overall Signal"},
        gauge={
            'axis': {'range': [1, 5], 'tickvals': [1, 2, 3, 4, 5], 
                    'ticktext': ['Strong Sell', 'Sell', 'Neutral', 'Buy', 'Strong Buy']},
            'steps': [
                {'range': [1, 2], 'color': 'red'},
                {'range': [2, 3], 'color': 'lightcoral'},
                {'range': [3, 4], 'color': 'gray'},
                {'range': [4, 5], 'color': 'lightgreen'},
                {'range': [5, 6], 'color': 'green'}
            ],
            'bar': {'color': overall_color}
        },
        number={'suffix': f" ({overall_text})"},
        delta={'reference': 3, 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}}
    ))
    
    # Update layout
    fig.update_layout(
        title="Technical Analysis Signal Dashboard",
        xaxis_title="Technical Indicators",
        yaxis_title="Signal Strength",
        yaxis=dict(
            tickvals=[1, 2, 3, 4, 5],
            ticktext=['Strong Sell', 'Sell', 'Neutral', 'Buy', 'Strong Buy'],
            range=[0.5, 5.5]
        ),
        template="plotly_white",
        height=500
    )
    
    return fig
