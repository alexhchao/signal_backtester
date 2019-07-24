
import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
sns.set()

pd.options.display.max_rows = 15
pd.options.display.max_columns = 15
pd.set_option("display.width",150)

from signalUnivariateStudy.SignalUnivariateStudy import SignalUnivariateStudy
from sklearn import linear_model

fwd_return_col = 'fwd_returns'

print('hello')

os.getcwd()
list_factors=['momentum','quality','growth','vol','value','size']

df = pd.read_csv('data/stock_data_actual_dates.csv').iloc[:,1:]

dates = df.date.unique()

date_generator = rolling_window(dates,
                   window_size = 12)

for date_range in date_generator:
    print(date_range)

    _df = df.query("date.isin(@date_range)", engine = 'python')

    _df


##################



reg = linear_model.LinearRegression()

X_y = _df.loc[:,list_factors + [fwd_return_col]]
X_y = X_y[X_y[fwd_return_col].notnull()]
X_y = X_y.apply(normalize).fillna(0)


X = X_y.loc[:,list_factors]
y = X_y.loc[:,fwd_return_col ]

reg.fit(X,y)

LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,
                 normalize=False)
reg.coef_



#############
# seasonal generator

def rolling_window(dates,
                   window_size,
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
    if end_offset is None:
        end = len(dates) - window_size
    else:
        end = len(dates) - (window_size + end_offset)

    while start <= end:
        yield dates[start:start + window_size]
        start += jump


####
dates = list(df.date.unique())
dates = pd.to_datetime(dates)

date_df = pd.DataFrame({'date':dates,
              'year':[x.strftime("%Y") for x in dates],
            'month':[x.strftime("%m") for x in dates],
              })

for dt in dates:
    print(dt)
    target_month = dt.strftime("%m")
    filtered_df = date_df.query("date <= @dt").query("month == @target_month")

    all_prev_dates_this_month = list(filtered_df.date.astype(str))
    # get all prev dates with same month

seasonal_date_generator = rolling_seasonal_window(dates, window_size =12,
                                                  min_window = 2)

i=1
for date_range in seasonal_date_generator:
    print(i)
    print(date_range)
    i +=1

date_generator = rolling_window(dates, window_size =12)

for date_range in date_generator:
    print(date_range)



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



dt = dates[1]
pd.to_datetime(dt).strftime("%m")
