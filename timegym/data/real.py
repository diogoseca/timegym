
import numpy as np
import pandas as pd

def get_stock(ticker, columns='Close'):
    import yfinance
    stock = yfinance.Ticker(ticker).history(period='max', auto_adjust=False)
    #stock = stock.drop(columns=['Adj Close'])
    return stock[columns]
