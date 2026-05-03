# ML-Driven Dynamic Portfolio Optimization

## Project Overview
This project evolves a traditional **Modern Portfolio Theory (MPT)** model from a static Microsoft Excel implementation into a dynamic, machine-learning-powered engine. By moving away from historical averages and utilizing **XGBoost** for return forecasting and **PostgreSQL** for feature engineering, the model achieves superior risk-adjusted returns.

### Key Evolution: From Static to Dynamic
- **The Problem:** Static optimization in Excel (using Solver) is prone to overfitting and "corner solutions," often resulting in 90%+ allocation to a single low-volatility asset.
- **The Solution:** A rolling-window Python pipeline that updates daily. It uses XGBoost to predict $r_t$ based on lagged returns, rolling volatility, and broader market trends (Nifty 50).

## Tech Stack
- **Database:** PostgreSQL (Window functions for log-return & feature engineering)
- **Machine Learning:** XGBoost Regressor
- **Optimization:** SciPy (SLSQP) for Sharpe Ratio maximization
- **Data Source:** Yahoo Finance (yfinance)


## How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Akshat-Singh-Kshatriya/ML-Portfolio-Optimization-and-Backtesting.git
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Engine

```bash
python data_extraction.py
python portfolio.py
```
