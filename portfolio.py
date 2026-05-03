import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

RISK_FREE_RATE = 0.07
WINDOW_SIZE = 126      

df = pd.read_csv('portfolio_features.csv', parse_dates=['Date'])
df.set_index('Date', inplace=True)
assets = df['Asset'].unique()

XGB_FEATURES = ['lag_1', 'lag_2', 'lag_5', 'rolling_vol_21d', 'market_return', 'market_lag_1']


def optimize_portfolio(expected_returns, cov_matrix, rf_rate):
    num_assets = len(expected_returns)

    def neg_sharpe(weights):
  
        port_return = np.sum(weights * expected_returns) * 252
  
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))

        if port_vol == 0:
            return 0
        return -(port_return - rf_rate) / port_vol

    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    bounds = tuple((0.05, 0.40) for _ in range(num_assets))
    
    init_guess = np.array(num_assets * [1. / num_assets])
    
    res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return res.x


unique_dates = df.index.unique().sort_values()
results = []


for t in range(WINDOW_SIZE, len(unique_dates) - 1):
    current_date = unique_dates[t]
    next_date = unique_dates[t + 1]
    

    train_df = df.loc[unique_dates[t - WINDOW_SIZE : t]]
    predicted_returns = []
  
    for asset in assets:
        asset_train = train_df[train_df['Asset'] == asset]
  
        X_train = asset_train[XGB_FEATURES]
        y_train = asset_train['target_return']
        
        X_now = df[(df.index == current_date) & (df['Asset'] == asset)][XGB_FEATURES]

        if X_now.empty or len(X_train) == 0:
            predicted_returns.append(0)
            continue
   
        model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.08, max_depth=3, verbosity=0)
        model.fit(X_train, y_train)

        predicted_returns.append(model.predict(X_now)[0])
    
    pivot_returns = train_df.pivot(columns='Asset', values='target_return').dropna()
    
    if pivot_returns.empty:
     
        weights = np.array([1. / len(assets)] * len(assets))
    else:
        cov_matrix = pivot_returns.cov().values

        weights = optimize_portfolio(np.array(predicted_returns), cov_matrix, RISK_FREE_RATE)

    actual_returns_df = df[df.index == next_date].set_index('Asset')
 
    actual_returns = actual_returns_df.reindex(assets)['target_return'].fillna(0).values
    
    daily_return = np.sum(weights * actual_returns)
    
    results.append({
        'Date': next_date,
        'Daily_Return': daily_return,
        'Weights': dict(zip(assets, weights))
    })

res_df = pd.DataFrame(results)
res_df['Cumulative_Return'] = (1 + res_df['Daily_Return']).cumprod()


ann_return = res_df['Daily_Return'].mean() * 252
ann_vol = res_df['Daily_Return'].std() * np.sqrt(252)
final_sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0


print(f"Annualized Return:      {ann_return:.2%}")
print(f"Annualized Volatility:  {ann_vol:.2%}")
print(f"Sharpe Ratio:           {final_sharpe:.2f}")



fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1.2, 1]})


ax1.plot(res_df['Date'], res_df['Cumulative_Return'], color='#2ca02c', linewidth=2)
ax1.set_title(f'Dynamic ML Portfolio Backtest w/ Nifty 50 (Sharpe: {final_sharpe:.2f})', fontsize=14, fontweight='bold')
ax1.set_ylabel('Growth of $1', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.fill_between(res_df['Date'], 1, res_df['Cumulative_Return'], 
                 where=(res_df['Cumulative_Return'] >= 1), color='#2ca02c', alpha=0.1)
ax1.fill_between(res_df['Date'], 1, res_df['Cumulative_Return'], 
                 where=(res_df['Cumulative_Return'] < 1), color='#d62728', alpha=0.1)

weights_plot_df = pd.json_normalize(res_df['Weights'])
weights_plot_df.index = res_df['Date']

ax2.stackplot(weights_plot_df.index, weights_plot_df.T, labels=assets, alpha=0.85)
ax2.set_title('Dynamic Asset Allocation (Capped at 40% Max Weight)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Weight Allocation (0 to 1)', fontsize=12)
ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11)
ax2.margins(x=0, y=0)
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()