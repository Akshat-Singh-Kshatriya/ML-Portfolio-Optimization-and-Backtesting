# Setup

import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from pypfopt import risk_models, expected_returns, EfficientFrontier, objective_functions
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Visual Settings
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
print("Libraries Installed Successfully.")

# Data Extraction
TICKERS = ['M&M.NS','SUNPHARMA.NS', 'DIVISLAB.NS','LT.NS','ICICIBANK.NS', 'HDFCBANK.NS','RELIANCE.NS']
BENCHMARK = '^NSEI'

START_DATE = '2018-01-01'
END_DATE = '2023-01-01'

def get_data(tickers):
    data = yf.download(tickers, start=START_DATE, end=END_DATE)['Close']

    # Clean & Calculate Log Returns
    data = data.dropna()
    log_returns = np.log(data / data.shift(1)).dropna()

    return data, log_returns

prices, log_returns = get_data(TICKERS)
benchmark_prices = yf.download(BENCHMARK, start=START_DATE, end=END_DATE)['Close']

print("Data Extracted:")
print(prices.tail(3))

# ML Optimising Engine
def create_features(price_series):
    df = pd.DataFrame({'Close': price_series.copy()})

    # 1. RSI (Momentum)
    rsi = RSIIndicator(close=df['Close'], window=14)
    df['RSI'] = rsi.rsi()

    # 2. MACD (Trend)
    macd = MACD(close=df['Close'])
    df['MACD'] = macd.macd_diff()

    # 3. Rolling Volatility
    df['Vol_20'] = df['Close'].pct_change().rolling(20).std()

    # TARGET: 1 if Next Day Return > 0 (Price goes UP)
    df['Target'] = (df['Close'].pct_change().shift(-1) > 0).astype(int)

    return df.dropna()

ml_predictions = {}

for ticker in TICKERS:
    # Prepare Data
    stock_df = create_features(prices[ticker])
    X = stock_df[['RSI', 'MACD', 'Vol_20']]
    y = stock_df['Target']

    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=5)
    accuracies = []

    # Train model using Cross Validation
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, eval_metric='logloss')

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))

    # Retrain on full data for final prediction
    model.fit(X, y)
    last_day_features = X.iloc[[-1]]
    prob_up = model.predict_proba(last_day_features)[0][1]
    ml_predictions[ticker] = prob_up

    avg_acc = np.mean(accuracies)
    print(f"   {ticker}: Validation Accuracy = {avg_acc:.2%} | Probability UP = {prob_up:.2%}")
selected_tickers = [t for t, prob in ml_predictions.items() if prob > 0.5]
print(f"\n Selected stocks for Portfolio: {selected_tickers}")

# Optimization: Sharpe Ratio
if not selected_tickers:
    print("Market Bearish. Hold Cash.")
    cleaned_weights = {t: 0 for t in TICKERS}
else:
    # 1. Expected Returns
    mu = expected_returns.mean_historical_return(prices[selected_tickers])

    # 2. Risk Model (Covariance)
    S = risk_models.CovarianceShrinkage(prices[selected_tickers]).ledoit_wolf()

    # 3. Optimize for Maximum Sharpe Ratio
    ef = EfficientFrontier(mu, S)
    ef.add_objective(objective_functions.L2_reg, gamma=1.0) # Advanced Regularization

    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    print("\n OPTIMIZED WEIGHTS:")
    weight_series = pd.Series(cleaned_weights).sort_values(ascending=False)
    print(weight_series[weight_series > 0])

    ef.portfolio_performance(verbose=True)

# Backtesting
initial_capital = 100000 # Rs

# Calculate Portfolio Returns
portfolio_prices = prices[list(cleaned_weights.keys())]
weighted_returns = portfolio_prices.pct_change().mul(list(cleaned_weights.values()), axis=1).sum(axis=1)

transaction_cost = 0.0005 # 0.05% per trade
net_returns = weighted_returns - transaction_cost

# Cumulative Returns
portfolio_curve = (1 + net_returns).cumprod() * initial_capital

# Benchmark Returns (Nifty 50)
if isinstance(benchmark_prices, pd.DataFrame):
    benchmark_prices = benchmark_prices.iloc[:, 0]
benchmark_returns = benchmark_prices.pct_change()
benchmark_curve = (1 + benchmark_returns).cumprod() * initial_capital

# Metrics
total_return = (portfolio_curve.iloc[-1] / initial_capital) - 1
cagr = ((portfolio_curve.iloc[-1] / initial_capital) ** (252/len(portfolio_prices))) - 1
drawdown = (portfolio_curve / portfolio_curve.cummax()) - 1
max_drawdown = drawdown.min()

print(f"\n Backtesting Results:")
print(f"Total Return: {total_return:.2%}")
print(f"CAGR (Annual Growth): {cagr:.2%}")
print(f"Max Drawdown: {max_drawdown:.2%}")

# Plots
# 1. Cumulative Returns Comparison
plt.figure(figsize=(12, 6))
portfolio_curve.plot(label='Optimized Portfolio', color='lime', linewidth=2)
benchmark_curve.plot(label='Nifty 50 Benchmark', color='gray', linestyle='--', alpha=0.7)
plt.title('Strategy vs Benchmark: Growth of Rs 100,000')
plt.ylabel('Portfolio Value (Rs)')
plt.legend()
plt.show()

# 2. Efficient Frontier (Risk vs Return)
# Re-calculate EF for plotting purposes
mu = expected_returns.mean_historical_return(prices[selected_tickers])
S = risk_models.CovarianceShrinkage(prices[selected_tickers]).ledoit_wolf()
ef = EfficientFrontier(mu, S, solver="ECOS") # Explicitly set solver to ECOS

from pypfopt import plotting # Added import statement
fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.title('Efficient Frontier (Risk vs Return Trade-off)')
plt.show()

# 3. Asset Correlation Heatmap (Risk Management)
plt.figure(figsize=(10, 8))
sns.heatmap(log_returns[selected_tickers].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Asset Correlation Matrix (Diversification Check)')
plt.show()

# 4. Underwater Plot (Drawdowns)
plt.figure(figsize=(12, 4))
drawdown.plot(color='red', alpha=0.6)
plt.fill_between(drawdown.index, drawdown, color='red', alpha=0.3)
plt.title('Underwater Plot: Drawdown Magnitude')
plt.ylabel('Loss from Peak')
plt.show()
