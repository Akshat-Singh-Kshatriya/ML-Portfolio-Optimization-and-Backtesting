import yfinance as yf
import pandas as pd

def download_data_for_postgres():
    tickers = ['INFY.NS', 'HDFCBANK.NS', 'LT.NS', 'M&M.NS']
    
    start_date = '2020-01-01'
    end_date = '2025-05-31'
    data = yf.download(tickers, start=start_date, end=end_date)['Close']
    data = data.reset_index()

    
    raw_prices = pd.melt(
        data,
        id_vars=['Date'],           
        value_vars=tickers,         
        var_name='Asset',           
        value_name='Close_Price'    
    )

    raw_prices = raw_prices.sort_values(by=['Date', 'Asset'])
    
    
    raw_prices = raw_prices.dropna(subset=['Close_Price'])

    file_name = 'raw_prices.csv'
    raw_prices.to_csv(file_name, index=False)
    

if __name__ == "__main__":
    download_data_for_postgres()
    
nifty = yf.download('^NSEI', start='2020-01-01', end='2025-05-31')['Close']
nifty = nifty.reset_index()
nifty.columns = ['Date', 'Nifty_Close']


nifty.to_csv('nifty50_prices.csv', index=False)