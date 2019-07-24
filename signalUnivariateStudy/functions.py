
import pandas as pd
import numpy as np
import os
import sys
from scipy.stats import zscore
from scipy.stats import norm

def check_if_matrix_has_nans(m):
    return np.any(np.isnan(m))


def normalize(x):
    """
    percentile rank than inverse normal dist

    Parameters
    ----------
    x
    Returns
    -------
    """
    ranks = x.rank()
    _x = ranks / (1 + max(ranks.dropna()))  # na messes things up
    return pd.Series(norm.ppf(_x))


def is_binary(col):
    """
    Parameters
    ----------
    col
    Returns
    -------
    """
    if not isinstance(col, pd.Series):
        raise ValueError("column is not a series! please try again")

    return set(col.unique()) == {0, 1}


def is_not_binary(col):
    """
    Parameters
    ----------
    col
    Returns
    -------
    """
    if not isinstance(col, pd.Series):
        raise ValueError("column is not a series! please try again")

    return not set(col.unique()) == {0, 1}

#def zscore(x):
#    """
#
#    Returns
#    -------
#
#    """
#    return (x - x.mean())/x.std()

def rolling_window(dates,
                   window_size,
                   start = 0,
                   end_offset = None,
                   jump = 1):
    """
    
    Parameters
    ----------
    dates
    window_size
    start
    end_offset
    jump

    Returns
    -------
    
    """
    if end_offset is None:
        end = len(dates) - window_size
    else:
        end = len(dates) - (window_size + end_offset)

    while start <= end:
        yield dates[start:start+window_size]
        start += jump


def rolling_seasonal_window(dates,
                   window_size,
                    min_window = None,
                   start=0,
                   end_offset=None,
                   jump=1):
    """

    Parameters
    ----------
    dates
    window_size
    start
    end_offset
    jump

    Returns
    -------

    """
    if min_window is None:
        min_window = window_size * 0.5

    if end_offset is None:
        end = len(dates) - window_size
    else:
        end = len(dates) - (window_size + end_offset)

    _dates = pd.to_datetime(dates)

    date_df = pd.DataFrame({'date': _dates,
                            'year': [x.strftime("%Y") for x in _dates],
                            'month': [x.strftime("%m") for x in _dates],
                            })

    while start < end:
        dt = _dates[start + window_size]
        target_month = dt.strftime("%m")
        filtered_df = date_df.query("date <= @dt").query("month == @target_month")
        length_window = filtered_df.shape[0]

        if length_window < min_window:
            start += jump
            continue # skip rest
        elif length_window > window_size:
            filtered_df = filtered_df.iloc[-window_size:]
        all_prev_dates_this_month = list(filtered_df.date.astype(str))
        yield all_prev_dates_this_month
        #yield dates[start:start + window_size]
        start += jump

def percentile_rank(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    return x.rank(pct=True)


def add_sector_neutral_column(df,
                              col_to_neutralize,
                              neutralized_col_name=None,
                              agg_col_names=['date', 'sector']):
    """

    Parameters
    ----------
    df
    agg_col_names

    Returns
    -------

    """
    _df = df.copy()

    if neutralized_col_name is None:
        neutralized_col_name = '{}_SN'.format(col_to_neutralize)

    _df[neutralized_col_name] = _df.groupby(['date', 'sector'])[col_to_neutralize].rank(pct=True)
    return _df



def calc_sharpe(x,
                n=12):
    """

    Parameters
    ----------
    x - pd.Series
    n = int, default = 12, adj fator

    Returns
    -------
    float
    """
    return x.mean() * np.sqrt(n) / x.std()


def calc_stats(_ret_series,
               n=12  # for monthly
               ):
    """

    Parameters
    ----------
    _ret_series

    Returns
    -------

    """
    _stats = {}

    adj_factor = np.sqrt(n)
    num_obs = _ret_series.shape[0]

    _stats['rets'] = _ret_series.mean() * n
    _stats['vol'] = _ret_series.std() * adj_factor
    _stats['sharpe'] = _stats['rets'] / _stats['vol']
    _stats['tstat'] = _stats['sharpe'] * np.sqrt(num_obs) / adj_factor
    _stats['start_dt'] = _ret_series.index[0]
    _stats['end_dt'] = _ret_series.index[-1]
    # import pdb; pdb.set_trace()
    return _stats



#############################
def add_quintiles_as_new_col(df,
                             col_name,
                             new_col_name = None,
                             groupby_col_name='date',
                             n=10):
    """
    add quantiles as new row to current df
    
    Parameters
    ----------
    df
    col_name
    new_col_name
    groupby_col_name
    n

    Returns
    -------

    """
    _df = df.copy()

    if new_col_name is None:
        new_col_name = '{}_q'.format(col_name)

    _df[new_col_name] = _df.groupby(groupby_col_name)[col_name].transform(lambda x: pd.qcut(
        x, q=n, labels= np.arange(1,n+1))).astype(str).apply(lambda x: x.split('.')[0])
    _df[new_col_name].replace('nan', np.NaN, inplace=True)

    return _df

# to do
# add percentile rank function
# add sector neutral functionality

