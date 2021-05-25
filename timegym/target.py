
import numpy as np
import pandas as pd

def multioutput_target(df, horizon=10, dropna=True):
    df = df.copy()
    for col in df.columns:
        for h in range(1, 1+horizon):
            df[f'{col}_t+{h}'] = df[col].shift(-h)
    if dropna:
        df.dropna(inplace=True)
    return df

def onestep_target(df, dropna=True):
    return multioutput_target(df, horizon=1, dropna=dropna)

def step_ahead_target(df, step=5, dropna=True):
    df = df.copy()
    for col in df.columns:
        df[f'{col}_t+{step}'] = df[col].shift(-step)
    if dropna:
        df.dropna(inplace=True)
    return df

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def fractional_weights(d, lags):
    # https://gist.github.com/skuttruf/fb82807ab0400fba51c344313eb43466
    # return the weights from the series expansion of the differencing operator
    # for real orders d and up to lags coefficients
    w = [1]    
    for k in range(1,lags):
        w.append(-w[-1]*((d-k+1))/k)
    return np.array(w).reshape(-1,1) 

def fractional_differencing(series, order:float, fractional_coefficients:int):
    # https://gist.github.com/skuttruf/fb82807ab0400fba51c344313eb43466
    assert order < fractional_coefficients, 'Accepting real orders order up to lag_cutoff coefficients'
    weights = fractional_weights(order, fractional_coefficients)
    res = 0
    for k in range(fractional_coefficients):
        res += weights[k] * series.shift(k).fillna(0)
    return res[fractional_coefficients:] 
    

def differencing(series, order=1, fractional_coefficients=10):
    # vanilla differencing
    if type(order) is int:
        return series.diff(order).iloc[order:]
    elif order.is_integer():
        return series.diff(int(order)).iloc[order:]
    # fractional differencing
    else:
        return fractional_differencing(series, order, fractional_coefficients)
