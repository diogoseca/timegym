
import numpy as np
import pandas as pd

def simple_moving_average(df, window_length=30, min_periods=1):
    df = pd.DataFrame(df)
    ma = df.iloc[:,0].rolling(window_length, min_periods=min_periods).mean()
    ma.rename(f'{df.columns[0]}_MA{window_length}', inplace=True)
    return pd.concat([df, ma], axis=1)

def exponential_moving_average(df, exponential_alpha=None, statistic='mean'):
    df = df.copy()
    df = df.ewm(alpha=exponential_alpha)
    if statistic=='mean':
        return df.mean()
    elif statistic=='median':
        return df.median()
    else:
        raise NotImplementedError()
