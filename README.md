# ML-Portfolio-Optimization-and-Backtesting

This project is an algorithmic portfolio construction engine built for **Nifty 50 equities**. It integrates **Machine Learning (XGBoost)** with **Mean-Variance Optimization** to generate dynamically rebalanced, risk-controlled portfolios. The framework prioritizes stability and risk-adjusted returns and was stress-tested during the 2020 COVID market crash.

## 🚀 Key Features

* **Predictive Modeling**
    * XGBoost classifier for regime detection
    * Walk-Forward Validation (TimeSeriesSplit)
    * No look-ahead bias

* **Risk Management**
    * Ledoit-Wolf Shrinkage Covariance
    * L2 Regularization (prevents over-concentration)
    * Sector diversification enforcement

* **Dynamic Allocation**
    * Efficient Frontier optimization via `PyPortfolioOpt`
    * Sharpe Ratio maximization

* **Realistic Backtesting**
    * **0.05% transaction costs** included
    * Slippage modeling & periodic rebalancing
    * 
## 📊 Backtest Results (2018–2023)

| Metric | Strategy | Benchmark (Nifty 50) |
| :--- | :--- | :--- |
| **Total Return** | **44.79%** | 73.39% |
| **CAGR** | **7.84%** | -- |
| **Max Drawdown** | **-45.44%** | -- |
| **Sharpe Ratio** | **0.72** | -- |

### 📈 Interpretation
The strategy intentionally sacrifices peak returns to reduce concentration risk.
**L2 Regularization** forces diversification across Pharma, Auto, and Banking sectors instead of overweighting momentum-heavy stocks (like Reliance) which carry higher variance.

**Result:**
* More stable allocation during downturns.
* Lower estimation error in covariance matrices.
* Improved risk-adjusted performance (Sharpe Ratio).
  
## 🛠 Tech Stack

* **Language:** Python
* **Data:** `yfinance` (Yahoo Finance)
* **Machine Learning:** `XGBoost`, `Scikit-Learn` (TimeSeriesSplit)
* **Optimization:** `PyPortfolioOpt` (Efficient Frontier, Ledoit-Wolf)
* **Viz:** `Matplotlib`, `Seaborn`

---

## 💻 How to Run

### 1. Clone Repository
## 💻 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/Akshat-Singh-Kshatriya/ML-Portfolio-Optimization-and-Backtesting.git
cd ML-Portfolio-Optimization-and-Backtesting
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Engine

```bash
python main.py
```
