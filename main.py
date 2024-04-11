import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calculate_stock_return(stock_symbol, start_date, end_date, initial_weight):
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
    stock_data['SMA_200'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Signal'] = np.where(stock_data['SMA_5'] > stock_data['SMA_200'], 1, 0)
    stock_data['PrevSignal'] = stock_data['Signal'].shift(1)
    stock_data['Return'] = stock_data['Close'].pct_change()
    stock_data['StrategyReturn'] = np.where(stock_data['PrevSignal'] == 1, stock_data['Return'], 0)
    stock_data['CumulativeReturn'] = (1 + stock_data['StrategyReturn']).cumprod()
    stock_data['WeightedReturn'] = stock_data['CumulativeReturn'] * initial_weight
    return stock_data[['Close', 'CumulativeReturn', 'WeightedReturn']]


def calculate_portfolio_return(stock_symbols, weights, start_date, end_date):
    portfolio_return = pd.DataFrame()
    invested_return = pd.DataFrame()

    for symbol, weight in zip(stock_symbols, weights):
        stock_return = calculate_stock_return(symbol, start_date, end_date, weight)
        portfolio_return[symbol] = stock_return['WeightedReturn']
        invested_return[symbol] = stock_return['Close'] * weight

    portfolio_return['PortfolioReturn'] = portfolio_return.sum(axis=1)
    invested_return['InvestedReturn'] = invested_return.sum(axis=1)

    # Slice the data to start after 200 days
    portfolio_return = portfolio_return.iloc[200:]
    invested_return = invested_return.iloc[200:]

    # Index both return series at 100 at the start
    portfolio_return['PortfolioReturn'] = (portfolio_return['PortfolioReturn'] / portfolio_return['PortfolioReturn'].iloc[0]) * 100
    invested_return['InvestedReturn'] = (invested_return['InvestedReturn'] / invested_return['InvestedReturn'].iloc[0]) * 100

    return portfolio_return['PortfolioReturn'], invested_return['InvestedReturn']


# Example usage
stock_symbols = ['qqq','soxx','TQQQ','USD','CVNA']
num_stocks = len(stock_symbols)
weights = [1 / num_stocks] * num_stocks
start_date = '2007-01-01'
end_date = '2024-12-12'

portfolio_return, invested_return = calculate_portfolio_return(stock_symbols, weights, start_date, end_date)

plt.figure(figsize=(12, 6))
plt.plot(portfolio_return, label='Portfolio Return (Crossover Strategy)')
plt.plot(invested_return, label='Invested Return')
plt.xlabel('Date')
plt.ylabel('Indexed Return (Base = 100)')
plt.title('Portfolio Return Comparison')
plt.legend()
plt.grid(True)
plt.show()