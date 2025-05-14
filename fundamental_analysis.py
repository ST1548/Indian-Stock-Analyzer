import yfinance as yf
import pandas as pd

def perform_fundamental_analysis(ticker):
    """
    Performs fundamental analysis on a stock by retrieving key financial metrics.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing fundamental analysis metrics
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Key fundamental metrics to extract
        metrics = [
            # Valuation metrics
            'trailingPE', 'forwardPE', 'pegRatio', 'priceToBook', 
            'priceToSalesTrailing12Months', 'enterpriseToRevenue', 'enterpriseToEbitda',
            
            # Profitability metrics
            'profitMargins', 'operatingMargins', 'grossMargins', 'returnOnAssets', 
            'returnOnEquity', 'earningsGrowth', 'revenueGrowth',
            
            # Balance sheet metrics
            'totalCash', 'totalCashPerShare', 'totalDebt', 'debtToEquity',
            'currentRatio', 'quickRatio',
            
            # Dividend metrics
            'dividendRate', 'dividendYield', 'payoutRatio',
            
            # Growth and estimates
            'earningsQuarterlyGrowth', 'revenueQuarterlyGrowth',
            'targetMeanPrice', 'targetHighPrice', 'targetLowPrice',
            'numberOfAnalystOpinions'
        ]
        
        # Extract metrics from info dictionary
        fundamental_data = {metric: info.get(metric) for metric in metrics}
        
        # Get financial statements if available
        try:
            # Get balance sheet
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                # Get most recent total assets and liabilities
                assets = balance_sheet.iloc[0, 0] if 'Total Assets' in balance_sheet.index else None
                liabilities = balance_sheet.iloc[1, 0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else None
                
                if assets is not None and liabilities is not None:
                    fundamental_data['assets_to_liabilities'] = assets / liabilities if liabilities != 0 else None
            
            # Get income statement
            income_stmt = stock.income_stmt
            if not income_stmt.empty:
                # Calculate revenue and earnings trends if available
                if 'Total Revenue' in income_stmt.index and len(income_stmt.columns) >= 2:
                    current_revenue = income_stmt.loc['Total Revenue', income_stmt.columns[0]]
                    prev_revenue = income_stmt.loc['Total Revenue', income_stmt.columns[1]]
                    fundamental_data['revenue_trend'] = (current_revenue - prev_revenue) / prev_revenue if prev_revenue != 0 else None
        except:
            # If any error occurs with financial statements, continue without them
            pass
            
        return fundamental_data
    
    except Exception as e:
        print(f"Error performing fundamental analysis for {ticker}: {e}")
        return {}

def evaluate_fundamental_strength(data):
    """
    Evaluates the fundamental strength of a stock based on key metrics.
    
    Args:
        data (dict): Dictionary containing fundamental metrics
        
    Returns:
        dict: Dictionary containing fundamental strength evaluation
    """
    strength = {
        'overall': 'neutral',
        'valuation': 'neutral',
        'profitability': 'neutral',
        'financial_health': 'neutral',
        'growth': 'neutral',
        'explanation': 'Not enough data to evaluate'
    }
    
    if not data:
        return strength
    
    try:
        # Evaluate valuation
        pe_score = 0
        if data.get('trailingPE') is not None:
            pe = data.get('trailingPE')
            if pe < 15:
                pe_score = 1  # Potentially undervalued
            elif pe > 30:
                pe_score = -1  # Potentially overvalued
                
        pb_score = 0
        if data.get('priceToBook') is not None:
            pb = data.get('priceToBook')
            if pb < 1.5:
                pb_score = 1  # Potentially undervalued
            elif pb > 4:
                pb_score = -1  # Potentially overvalued
                
        valuation_score = pe_score + pb_score
        
        if valuation_score > 0:
            strength['valuation'] = 'positive'
        elif valuation_score < 0:
            strength['valuation'] = 'negative'
        
        # Evaluate profitability
        profit_score = 0
        if data.get('profitMargins') is not None:
            profit = data.get('profitMargins')
            if profit > 0.1:  # 10% profit margin
                profit_score = 1
            elif profit < 0.05:
                profit_score = -1
                
        roe_score = 0
        if data.get('returnOnEquity') is not None:
            roe = data.get('returnOnEquity')
            if roe > 0.15:  # 15% ROE
                roe_score = 1
            elif roe < 0.08:
                roe_score = -1
                
        profitability_score = profit_score + roe_score
        
        if profitability_score > 0:
            strength['profitability'] = 'positive'
        elif profitability_score < 0:
            strength['profitability'] = 'negative'
        
        # Evaluate financial health
        debt_score = 0
        if data.get('debtToEquity') is not None:
            debt = data.get('debtToEquity')
            if debt < 50:  # Low debt
                debt_score = 1
            elif debt > 100:  # High debt
                debt_score = -1
                
        liquidity_score = 0
        if data.get('currentRatio') is not None:
            cr = data.get('currentRatio')
            if cr > 1.5:  # Good liquidity
                liquidity_score = 1
            elif cr < 1:
                liquidity_score = -1
                
        health_score = debt_score + liquidity_score
        
        if health_score > 0:
            strength['financial_health'] = 'positive'
        elif health_score < 0:
            strength['financial_health'] = 'negative'
        
        # Evaluate growth
        growth_score = 0
        if data.get('earningsGrowth') is not None:
            earnings_growth = data.get('earningsGrowth')
            if earnings_growth > 0.1:  # 10% growth
                growth_score = 1
            elif earnings_growth < 0:
                growth_score = -1
                
        revenue_score = 0
        if data.get('revenueGrowth') is not None:
            revenue_growth = data.get('revenueGrowth')
            if revenue_growth > 0.1:  # 10% growth
                revenue_score = 1
            elif revenue_growth < 0:
                revenue_score = -1
                
        growth_score = growth_score + revenue_score
        
        if growth_score > 0:
            strength['growth'] = 'positive'
        elif growth_score < 0:
            strength['growth'] = 'negative'
        
        # Calculate overall score
        total_score = valuation_score + profitability_score + health_score + growth_score
        
        if total_score >= 3:
            strength['overall'] = 'strong'
        elif total_score > 0:
            strength['overall'] = 'positive'
        elif total_score == 0:
            strength['overall'] = 'neutral'
        elif total_score > -3:
            strength['overall'] = 'negative'
        else:
            strength['overall'] = 'weak'
        
        # Generate explanation
        explanation = []
        
        if strength['valuation'] == 'positive':
            explanation.append("Valuation metrics indicate the stock may be undervalued.")
        elif strength['valuation'] == 'negative':
            explanation.append("Valuation metrics indicate the stock may be overvalued.")
            
        if strength['profitability'] == 'positive':
            explanation.append("The company shows strong profitability metrics.")
        elif strength['profitability'] == 'negative':
            explanation.append("The company's profitability metrics are below industry averages.")
            
        if strength['financial_health'] == 'positive':
            explanation.append("The company has a strong financial position with manageable debt.")
        elif strength['financial_health'] == 'negative':
            explanation.append("The company's financial health shows some concerning indicators.")
            
        if strength['growth'] == 'positive':
            explanation.append("The company is showing healthy revenue and earnings growth.")
        elif strength['growth'] == 'negative':
            explanation.append("The company's growth metrics are below expectations.")
            
        strength['explanation'] = " ".join(explanation) if explanation else "Analysis based on limited data."
        
        return strength
        
    except Exception as e:
        print(f"Error evaluating fundamental strength: {e}")
        return strength
