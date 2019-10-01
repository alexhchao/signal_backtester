
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

print('hello')

os.getcwd()
list_factors=['sector', 'momentum','quality','growth','vol','value','size']

df = pd.read_csv('data/stock_data_actual_dates.csv').iloc[:,1:]

#df.groupby('date').count().plot()

#############################
# first build a light weight backtester
#############################
# start here
############################


sig = SignalUnivariateStudy(data_df = df,
                            factor_name = 'momentum',
                            #neutralizer_column = 'sector',
                            order = 'asc',
                            start_dt = '2008-01-31',
                            end_dt = '2014-12-31',
                            n = 5)

#sig.IC_time_series.plot(kind='bar')

sharpes = list(sig.stats.loc['sharpe'].values)
sharpes

mom_q5_sharpes = [0.4335993828993728, 0.629807334232841, 0.736147030627279,
       0.7806338800155603, 0.7417751759348701, 0.08860064205960581]


assert(sharpes == mom_q5_sharpes)

sig.data_df.groupby(['date','sector']).vol_SN.describe()

sig.wealth.plot()

sig

df['fwd_returns'].mean()

print(sig.stats)

pd.DataFrame(sig.stats)

#################

sig = SignalUnivariateStudy(data_df = df,
                            factor_name = 'fwd_returns',
                           # neutralizer_column = 'sector',
                            order = 'asc',
                            n = 10)

sig.stats.loc['rets',].plot()

sig.rets.loc[:,['LS - 5 minus 1']] .min()

#sig.rets.loc[:,['1']] .max()

############################
# run backtests on all factors

all_bts = {f:SignalUnivariateStudy(data_df = df,
                            factor_name = f,
                            neutralizer_column = 'sector',
                            order = 'asc',
                            n = 5) for f in list_factors[1:]}

#all_bts['vol'].stats.iloc[:,-1]

all_bts_stats = pd.concat({f:all_bts[f].stats.iloc[:,-1] for f in list_factors[1:]},axis=1)
all_bts_stats

df

all_bts['size'].wealth.plot()

list_factors = ['sector', 'momentum', 'quality', 'growth', 'vol', 'value', 'size']

df = pd.read_csv('../data/stock_data_actual_dates.csv').iloc[:, 1:]

sig = SignalUnivariateStudy(data_df=df,
                            factor_name='momentum',
                            neutralizer_column='sector',
                            order='asc',
                            n=5)
print(sig.stats)

sharpes = list(sig.stats.loc['sharpe'].values)

mom_q5_sharpes = [0.4335993828993728, 0.629807334232841, 0.736147030627279,
                  0.7806338800155603, 0.7417751759348701, 0.08860064205960581]

assert (sharpes == mom_q5_sharpes)

self.assertEqual(sharpes, mom_q5_sharpes,
                 "sharpes for momentum q5 sector neutral dont match!")




############################
# lets start building ML framework

import statsmodels.api as sm

y_var = 'fwd_returns'
neutralizer_col = 'sector'

# Generate artificial data (2 regressors + constant)
# exclude where y is null


_df = _df.query("date <= '2006-12-31'")

_df = df.copy()


_df = df[df[y_var].notnull()]

y = _df['fwd_returns']

X = _df.loc[:,list_factors]

# sector neutral zscore
X_z = X.groupby(neutralizer_col).apply(zscore)

X_z = X_z.loc[:,[x for x in list_factors if x != neutralizer_col]]

X_z.fillna(0)


X_z = sm.add_constant(X_z)

# Fit regression model
results = sm.OLS(y, X_z.fillna(0)).fit()

# Inspect the results
print(results.summary())

###########################################

ols_results = run_ols(df = df,
        y_var = 'fwd_returns',
        neutralizer_col = 'sector',
        list_predictor_cols = ['sector', 'momentum', 'quality', 'growth', 'vol', 'value', 'size'])


ols_results



# run rolling ols
###########################################

list_dts = df.date.unique()

window = 36

rolling_model = {}
for i in np.arange(window ,len(list_dts)):
    start_dt = list_dts[i-window ]
    end_dt = list_dts[i]
    print('start = {}, end = {}'.format(start_dt, end_dt))

    ols_results = run_ols(df=df.query("date >= @start_dt").query("date <= @end_dt"),
                          y_var='fwd_returns',
                          neutralizer_col='sector',
                          list_predictor_cols=['sector', 'momentum', 'quality', 'growth', 'vol', 'value', 'size'])
    rolling_model[end_dt] = ols_results

rolling_model[end_dt].summary()

rolling_model[end_dt].params

rolling_model[end_dt].rsquared

rolling_model[end_dt].pvalues

t_stats = rolling_model[end_dt].params/rolling_model[end_dt].bse

all_ts = {}
all_rsqrs = {}
for dt in rolling_model.keys():
    all_ts[dt] = rolling_model[dt].params/rolling_model[dt].bse
    all_rsqrs[dt] = rolling_model[dt].rsquared


pd.DataFrame(all_ts).T.plot()

pd.Series(all_rsqrs).plot()
###########################################









###########################################


help(sm.RegressionResults)

#help(sm.OLS)
#help(sm.OLS.model)

#####
# params
####
#
def run_ols(df,
            y_var,
            neutralizer_col,
            list_predictor_cols,
            neutralizer_func = zscore,
            fill_na_with = 0):
    """

    Parameters
    ----------
    y_var
    neutralizer_col
    list_predictor_cols
    neutralizer_func

    Returns
    -------

    """
    _df = df.copy()

    _df = _df[_df[y_var].notnull()] # exclude where y is null
    y = _df[y_var]
    X = _df.loc[:, list_factors]

    # sector neutral zscore
    X_z = X.groupby(neutralizer_col).apply(zscore)

    X_z = X_z.loc[:, [x for x in list_factors if x != neutralizer_col]]

    X_z = sm.add_constant(X_z)

    # Fit regression model
    results = sm.OLS(y, X_z.fillna(fill_na_with)).fit()

    # Inspect the results
    print(results.summary())

    return results









def zscore(x):
    return (x-x.mean() )/ x.std()



############################

def calc_stats(_ret_series,
               n=12 # for monthly
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
    _stats['sharpe']=_stats['rets'] /_stats['vol']
    _stats['tstat'] = _stats['sharpe'] * np.sqrt(num_obs) / adj_factor
    _stats['start_dt'] = _ret_series.index[0]
    _stats['end_dt'] = _ret_series.index[-1]
    #import pdb; pdb.set_trace()
    return _stats




#############################

def add_quintiles_as_new_col(df,
                             col_name,
                             new_col_name,
                             groupby_col_name = 'date',
                             n=10):
    """
    
    :param col_name: 
    :param n: 
    :param new_col_name: 
    :param groupby_col_name: 
    :return: 
    """
    _df = df.copy()

    _df[new_col_name] = _df.groupby(groupby_col_name)[col_name].transform(lambda x: pd.qcut(
        x, q = n, labels = np.arange(1,n+1), precision=0)).astype(str)
    return _df



#############################
#
#############################





#############################
#
#############################






#############################
#
#############################





#############################
#
#############################



