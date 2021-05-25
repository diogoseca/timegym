
import numpy as np
import pandas as pd

def add_lagged_target(df, lags=10, dropna=True):
    df = df.copy()
    for col in df.columns:
        for lag in range(1, 1+lags):
            df[f'{col}_t-{lag}'] = df[col].shift(lag)
    if dropna:
        df.dropna(inplace=True)
    return df

def cyclical_encoding(values, scaled=False, min_value=None, max_value=None):
    if scaled:
        scaled = values
    else:
        min_value = min_value or np.min(values)
        max_value = max_value or np.max(values)
        scaled = (values-min_value)/max_value
    cos = np.cos(scaled*np.pi*2)
    sin = np.sin(scaled*np.pi*2)
    return cos, sin

def add_time_features(df):
    df = df.copy()
    df['time'] = df.index.astype(int)
    df['time'] = df['time'] / df['time'].std()
    if df.index.year.nunique()>1: df['year'] = df.index.year
    # cyclical encoding using trigonometric functions (sin, cos)
    cyclical_time_features = True
    if cyclical_time_features:    
        if df.index.quarter.nunique()>1: 
            cos, sin = cyclical_encoding(df.index.quarter, min_value=1, max_value=4)
            df['quarter_cos'], df['quarter_sin'] = cos, sin
        if df.index.month.nunique()>1: 
            cos, sin = cyclical_encoding(df.index.month, min_value=1, max_value=12)
            df['month_cos'], df['month_sin'] = cos, sin
        unadjusted_days = True
        if unadjusted_days and df.index.day.nunique()>1:
            cos, sin = cyclical_encoding(df.index.day, min_value=1, max_value=31)
            df['day_cos'], df['day_sin'] = cos, sin
        adjusted_days = True
        if adjusted_days and df.index.day.nunique()>1:
            adjusted_days = (df.index.day - 1) / df.index.daysinmonth
            cos, sin = cyclical_encoding(adjusted_days, scaled=True)
            df['day_adj_cos'], df['day_adj_sin'] = cos, sin
        if df.index.weekday.nunique()>1: 
            cos, sin = cyclical_encoding(df.index.weekday, min_value=0, max_value=6)
            df['weekday_cos'], df['weekday_sin'] = cos, sin
        if df.index.hour.nunique()>1:
            cos, sin = cyclical_encoding(df.index.hour, min_value=0, max_value=23)
            df['hour_cos'], df['hour_sin'] = cos, sin
        if df.index.minute.nunique()>1:
            cos, sin = cyclical_encoding(df.index.minute, min_value=0, max_value=59)
            df['minute_cos'], df['minute_sin'] = cos, sin
        if df.index.second.nunique()>1:
            cos, sin = cyclical_encoding(df.index.second, min_value=0, max_value=59)
            df['second_cos'], df['second_sin'] = cos, sin
        if df.index.microsecond.nunique()>1:
            cos, sin = cyclical_encoding(df.index.miliosecond, min_value=0, max_value=999999)
            df['microsecond_cos'], df['microsecond_sin'] = cos, sin
    # no encoding
    else:
        if df.index.quarter.nunique()>1: df['quarter'] = df.index.quarter
        if df.index.month.nunique()>1: df['month'] = df.index.month
        if df.index.day.nunique()>1: df['day'] = df.index.day
        if df.index.weekday.nunique()>1: df['weekday'] = df.index.weekday
        if df.index.hour.nunique()>1: df['hour'] = df.index.hour
        if df.index.minute.nunique()>1: df['minute'] = df.index.minute
        if df.index.second.nunique()>1: df['second'] = df.index.second
        if df.index.microsecond.nunique()>1: df['microsecond'] = df.index.microsecond
    return df

import tsfel
from tqdm import tqdm

def add_tsfel_features(df, use_cols=None, window_size=100, domain=None, frequency=100, dropna=True, show_progress=False, lag=1):
    if use_cols is None:
        data = df
    else: 
        data = df[use_cols]

    cfg = tsfel.get_features_by_domain(domain=domain)
    
    features = []
    iterator = range(window_size, df.shape[0]-lag)
    if show_progress: iterator = tqdm(iterator)
    for window_end in iterator:
        window_start = window_end - window_size
        window = data.iloc[window_start:window_end]
        features.append(tsfel.time_series_features_extractor(cfg, window, verbose=0, fs=frequency))
    features = pd.concat(features, ignore_index=True)
    features.index = df.index[window_size+lag:]
        
    df = pd.concat([data, features], axis=1)
    if dropna:
        df.dropna(inplace=True)
    return df
