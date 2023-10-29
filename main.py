import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Calculate the historical volatility of a stock based on daily returns
def calculate_volatility(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change()
    volatility = stock_data['Daily_Return'].std() * (252 ** 0.5)
    return volatility

# Calculate the alpha and beta values of a stock relative to a market index
def calculate_alpha_and_beta(ticker, market_index, risk_free_rate, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    market_data = yf.download(market_index, start = start_date, end = end_date)
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    market_returns = market_data['Adj Close'].pct_change().dropna()
    excess_returns = stock_returns - risk_free_rate
    excess_market_returns = market_returns - risk_free_rate
    beta = np.cov(excess_returns, excess_market_returns)[0, 1] / np.var(excess_market_returns)
    alpha = excess_returns.mean() - beta * excess_market_returns.mean()
    return alpha, beta

# Calculate the Sharpe Ratio of a stock, indicating its risk-adjusted performance
def calculate_sharpe_ratio(ticker, risk_free_rate, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    excess_returns = stock_returns - risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    return sharpe_ratio

# Calculate the maximum drawdown of a stock
def calculate_max_drawdown(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    cumulative_returns = (1 + stock_data['Adj Close'].pct_change()).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns / peak) - 1
    max_drawdown = drawdown.min()
    return max_drawdown

# Calculate the Treynor Ratio of a stock relative to a market index
def calculate_treynor_ratio(ticker, market_index, risk_free_rate, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    excess_returns = stock_returns - risk_free_rate
    _, beta = calculate_alpha_and_beta(ticker, [market_index], risk_free_rate, start_date, end_date)
    treynor_ratio = excess_returns.mean() / beta
    return treynor_ratio

# Calculate the Value at Risk (VaR) of a stock
def calculate_var(stock_returns, alpha = 0.05):
    return np.percentile(stock_returns, alpha * 100)

# Calculate the Calmar Ratio of a stock, measuring risk-adjusted performance over a specific period
def calculate_calmar_ratio(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start = start_date, end = end_date)
    stock_returns = stock_data['Adj Close'].pct_change().dropna()
    annualized_return = ((1 + stock_returns.mean()) ** 252) - 1
    max_drawdown = calculate_max_drawdown(ticker, start_date, end_date)
    calmar_ratio = annualized_return / abs(max_drawdown)
    return calmar_ratio

# Calculate the downside deviation of a stock's returns
def calculate_downside_deviation(stock_returns, threshold = 0):
    downside_returns = stock_returns[stock_returns < threshold]
    downside_deviation = np.std(downside_returns)
    return downside_deviation

# Calculate the tracking error between a stock's returns and benchmark returns
def calculate_tracking_error(stock_returns, benchmark_returns):
    return np.std(stock_returns - benchmark_returns)

# Calculate the trend of historical returns
def calculate_trend(stock_returns):
    trend = np.polyfit(range(len(stock_returns)), stock_returns, 1)[0]
    return trend * (10 ** 6)

# Make a final rating of safety of the stock
def generate_risk_assessment(ticker, market_indices, risk_free_rate, start_date, end_date):
    volatility = calculate_volatility(ticker, start_date, end_date)
    sharpe_ratio = calculate_sharpe_ratio(ticker, risk_free_rate, start_date, end_date)
    max_drawdown = calculate_max_drawdown(ticker, start_date, end_date)
    treynor_ratios = [calculate_treynor_ratio(ticker, index, risk_free_rate, start_date, end_date) for index in market_indices]
    calmar_ratio = calculate_calmar_ratio(ticker, start_date, end_date)
    var = calculate_var(yf.download(ticker, start = start_date, end = end_date)['Adj Close'].pct_change().dropna(), alpha = 0.05)
    downside_deviation = calculate_downside_deviation(yf.download(ticker, start = start_date, end = end_date)['Adj Close'].pct_change().dropna())
    trend = calculate_trend(yf.download(ticker, start = start_date, end = end_date)['Adj Close'].pct_change().dropna())
    weights = {
        'volatility': 0.35,
        'sharpe_ratio': 0.1,
        'max_drawdown': 0.1,
        'treynor_ratio': 0.1,
        'calmar_ratio': 0.1,
        'var': 0.05,
        'downside_deviation': 0.05,
        'trend': 0.15
    }
    risk_score = (
        weights['volatility'] * (1 - (volatility / 100)) +
        weights['sharpe_ratio'] * sharpe_ratio +
        weights['max_drawdown'] * (1 - max_drawdown) +
        weights['treynor_ratio'] * np.mean(treynor_ratios) +
        weights['calmar_ratio'] * calmar_ratio +
        weights['var'] * (1 - (var / 100)) +
        weights['downside_deviation'] * (1 - (downside_deviation / 100)) +
        weights['trend'] * (1 - (abs(trend) / 100))
    ) * 100
    risk_score = max(1, min(100, risk_score))
    print(f"\nVolatility: {volatility:.2%}")
    print(f"\nMax Drawdown: {max_drawdown:.2%}")
    print(f"\nValue at Risk (VaR): {var:.2%}")
    print(f"\nDownside Deviation: {downside_deviation:.2%}")
    print(f"\nTrend: {trend:.2f}%")
    print(f"\nSharpe Ratio: {sharpe_ratio:.4f}")
    print(f"\nCalmar Ratio: {calmar_ratio:.4f}")
    print("\nTreynor Ratios:")
    for i, index in enumerate(market_indices):
        print(f" {index}: {treynor_ratios[i]:.4f}")
    return risk_score

# Get user input from terminal and print results
ticker_symbol = input("Enter the stock symbol (for instance, AAPL): ")
start_date = (datetime.now() - timedelta(days = 366)).strftime('%Y-%m-%d')
end_date = (datetime.now() - timedelta(days = 1)).strftime('%Y-%m-%d')
market_indices = ['^GSPC', '^DJI', '^IXIC']
risk_free_rate = 0.02
risk_assessment = generate_risk_assessment(ticker_symbol, market_indices, risk_free_rate, start_date, end_date)
print(f"\nWith 1 being the least safe and 100 being the most safe, {ticker_symbol}'s safety score in terms of risk of investment is {risk_assessment:.2f}/100. This is based on data from the past year.")
