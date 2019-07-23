
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




