

import numpy as np
import pandas as pd
import datetime
from natsort import natsorted
from signalUnivariateStudy.functions import calc_stats, add_quintiles_as_new_col, add_sector_neutral_column


def computeMaxDrawDown(wealth,
                       max_only=True):
    """
    compute max drawdown given a wealth curve

    Parameters
    ----------
    wealth - pd.Series
    max_only - boolean, if True, return maxDD, else return time series drawdowns

    Returns
    -------
    float
    """
    drawdown = 1 - wealth.div(wealth.cummax())
    #return drawdown.max(axis=0)
    if max_only:
       return drawdown.max(axis=0)
    else:
       return drawdown

def getAnnualizationFactor(freq):
    """
    
    Parameters
    ----------
    freq

    Returns
    -------

    """
    if freq == 'month':
        return 12
    elif freq == 'day':
        return 252
    elif freq == 'week':
        return 52
    elif freq == 'quater':
        return 4
    else:
        return 1

def getFrequency(list_dates):
    """
    get frequency from a list of dates
    
    Parameters
    ----------
    list_dates

    Returns
    -------
    string (day, month, quater, week, year)
    
    """
    d = pd.Series(list_dates).diff().mean()

    d0 = {'day':datetime.timedelta(days=1),
          'week': datetime.timedelta(days=7),
          'month': datetime.timedelta(days=30),
          'quarter': datetime.timedelta(days=90),
          'year': datetime.timedelta(days=365)
          }
    d0 = pd.DataFrame(d0, index = [0]).T

    out = abs(d - d0).sort_values(by=0).index[0]
    return out



class Signal(object):
    """
    
    """
    def __init__(self,
                 df,
                 name = 'unnamed_factor',
                 direction = 'asc',
                 date_col_name = 'date',
                 stock_col_name = 'stock'):
        """
        
        Parameters
        ----------
        df
        name
        direction
        date_col_name
        stock_col_name
        """
        self.signal = df
        self.name = name
        self.direction = direction
        self.date_col_name = date_col_name
        self.stock_col_name = stock_col_name

    @property
    def dates(self):
        return pd.to_datetime(self.signal.index)

    @property
    def coverage(self):
        return self.signal.count(axis=1)

    @property
    def median_coverage(self):
        return self.signal.count(axis=1).median()

    @property
    def start_date(self):
        return self.dates[0].strftime("%Y-%m-%d")

    @property
    def end_date(self):
        return self.dates[-1].strftime("%Y-%m-%d")

    @property
    def freq(self):
        return getFrequency(self.dates)

    def __repr__(self):
        return("""
        [Signal]
        {}
        -------------------------------------
        Dates: {} dates, from {} to {}, every {}
        Median Coverage: {} stocks
        Direction: {}
        """.format(
            self.name,
            self.signal.shape[0],
            self.start_date,
            self.end_date,
            self.freq,
            self.median_coverage,
            self.direction
        ))


class Rets(Signal):

    #def __init__(self,
    #             df):
    #    super().__init__(df)

    @property
    def returns(self):
        return self.signal

    @returns.setter
    def returns(self, df):
        self.signal = df

    def __repr__(self):
        return("""
        [Rets]
        {}
        -------------------------------------
        Dates: {} dates, from {} to {}, every {}
        Median Coverage: {} stocks
        """.format(
            self.name,
            self.signal.shape[0],
            self.start_date,
            self.end_date,
            self.freq,
            self.median_coverage,
        ))



class Constituents(Signal):

    def __init__(self,
                 df):
        super().__init__(df)
        self.const = df
        print("setting constituents to be 1 or Null")
        self.signal[self.signal.notnull()] = 1.0

    #@property
    #def const(self):
    #    return self.signal

    #@const.setter
    #def const(self, df):
    #    self.signal = df
    #    print("setting constituents to be 1 or Null")
    #    self.signal[self.signal.notnull()] = 1.0

    def __repr__(self):
        return("""
        [Constituents]
        {}
        -------------------------------------
        Dates: {} dates, from {} to {}, every {}
        Median Coverage: {} stocks
        """.format(
            self.name,
            self.signal.shape[0],
            self.start_date,
            self.end_date,
            self.freq,
            self.median_coverage,
        ))


class Portfolio(Signal):

    def __init__(self,
                 df,
                 weight_scheme='equal'):
        super().__init__(df)
        self.weights = df
        self.weight_scheme = weight_scheme

    def __repr__(self):
        return("""
        [Portfolio]
        {}
        -------------------------------------
        Dates: {} dates, from {} to {}, every {}
        Median Coverage: {} stocks
        Weight Scheme: {}
        """.format(
            self.name,
            self.weights.shape[0],
            self.start_date,
            self.end_date,
            self.freq,
            self.median_coverage,
            self.weight_scheme
        ))


class SignalUnivariateStudy(object):
    """
    
    """

    def __init__(self,
                 data_df,
                 factor_name,
                 stock_col_name,
                 neutralizer_column = None,
                 order = 'asc',
                 n = 5,
                 date_col_name = 'date',
                 fwd_return_col_name = 'fwd_returns',
                 start_dt = None,
                 end_dt = None,
                 sector = None):
        """
        object that allows for backtesting of signals using L/S quantile portfolios 
        given a dataframe of [date, returns, factor_name, ...]

        Parameters
        ----------
        data_df - pd.DataFrame [date, returns, factor_name, ...]
        factor_name - name of factor to backtest
        order - 'asc' for ascending order (high - low) or 'desc' for descending (low - high)
        n - number of buckets, 5 = quintiles, 10 = deciles
        date_col_name - date column name, default = 'date'
        fwd_return_col_name - forward returns column name, default = 'fwd_returns'
        start_dt - date to start backtest
        end_dt - date to end backtest
        """

        self.data_df = data_df.copy()
        self.factor_name = factor_name
        self.order = order
        self.n = n
        self.date_col_name = date_col_name
        self.fwd_return_col_name = fwd_return_col_name
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.sector = sector
        #self.stats = None
        self.neutralizer_column = neutralizer_column
        self.stock_col_name = stock_col_name

        # fix fwd returns
        if self.data_df[fwd_return_col_name].mean() > 0.99:
            self.data_df[fwd_return_col_name] = self.data_df[fwd_return_col_name] * 0.01

        self.run_backtest()

    @property
    def dates(self):
        return list(pd.to_datetime(self.data_df[self.date_col_name].unique()))

    @property
    def freq(self):
        return getFrequency(self.dates)

    @property
    def annualizationFactor(self):
        return getAnnualizationFactor(self.freq)

    def run_backtest(self):
        """
        run backtest, calculate portfolios / returns for each quantile including long/short portfolio

        """

        #_df = self.data_df.copy()

        if self.start_dt is not None:
            _start = self.start_dt
            print("starting backtest on {}".format(_start))
            #self.data_df = self.data_df.query("date >= @_start ")
            #import pdb; pdb.set_trace()
            self.data_df = self.data_df[self.data_df[self.date_col_name] >= _start]

        if self.end_dt is not None:
            _end = self.end_dt
            print("ending backtest on {}".format(_end))
            #self.data_df = self.data_df.query("date <= @_end ")
            self.data_df = self.data_df[self.data_df[self.date_col_name] <= _end]

        if self.neutralizer_column is not None:
            print("neutralizing factor = {} using {}".format(self.factor_name,
                                                             self.neutralizer_column))
            self.data_df = self.add_sector_neutral_column(df=self.data_df,
                                       col_to_neutralize= self.factor_name,
                                       neutralized_col_name= None,
                                       agg_col_names=[self.date_col_name,
                                                      self.neutralizer_column])


            df_2 = self.add_quintiles_as_new_col(df = self.data_df,
                                            col_name= '{}_SN'.format(self.factor_name),
                                            new_col_name=None,
                                            groupby_col_name=self.date_col_name,
                                            n=self.n)
            self.factor_col_q_name = '{}_SN_q'.format(self.factor_name)
        else:
            df_2 = self.add_quintiles_as_new_col(df=self.data_df,
                                            col_name='{}'.format(self.factor_name),
                                            new_col_name=None,
                                            groupby_col_name=self.date_col_name,
                                            n=self.n)

            self.factor_col_q_name = '{}_q'.format(self.factor_name)

        #import pdb;
        #pdb.set_trace()
        #df_2.groupby('date').mom_q.value_counts()
        self.rets = df_2.groupby([
            self.factor_col_q_name, self.date_col_name])[
            self.fwd_return_col_name].mean().unstack().T.shift(1)

        # Long minus short basket
        if self.order == 'asc':
            self.rets['LS - {} minus {}'.format(self.n,1)] = self.rets[str(self.n)]-self.rets['1']
        else:
            self.rets['LS - {} minus {}'.format(1,self.n)] = self.rets['1'] - self.rets[str(self.n)]

        # okay this gets stats for each basket
        #self.stats = pd.DataFrame(self.rets.apply(lambda x: pd.Series(calc_stats(x))))

        self.wealth = np.cumprod(1 + self.rets)
        self.wealth.iloc[0, :] = 1  # start off at 1

    @staticmethod
    def calc_stats(_ret_series,
                   n=12  # for monthly
                   ):
        """
        calculate stats for return series 

        Parameters
        ----------
        _ret_series - pd.Series of returns

        Returns
        -------
        pd.DataFrame of stats

        """
        _stats = {}

        adj_factor = np.sqrt(n)
        num_obs = _ret_series.shape[0]

        _wealth = np.cumprod(1 + _ret_series)
        _wealth.iloc[0] = 1  # start off at 1

        _stats['returns'] = _ret_series.mean() * n
        _stats['volatility'] = _ret_series.std() * adj_factor
        _stats['sharpe'] = _stats['returns'] / _stats['volatility']
        _stats['tstat'] = _stats['sharpe'] * np.sqrt(num_obs) / adj_factor
        _stats['start_dt'] = _ret_series.index[0]#.strftime('%Y-%m-%d')
        _stats['end_dt'] = _ret_series.index[-1]#.strftime('%Y-%m-%d')
        _stats['maxDD'] = computeMaxDrawDown(wealth = _wealth)

        #_stats['freq'] = getFrequency(pd.to_datetime(_ret_series.index.unique()))

        # to add
        #################
        # avg_num_stocks
        # turnover
        #################

        #import pdb;pdb.set_trace()
        return _stats


    @property
    def stats(self):
        """
        get trading stats for the backtest 

        Returns
        -------

        """
        # this gets stats for each basket
        _stats = pd.DataFrame(self.rets.apply(lambda x: pd.Series(self.calc_stats(x,
                                                                                  self.annualizationFactor))))

        _stats_T = _stats.T
        _stats_T['order'] = self.order
        _stats_T['n_buckets'] = self.n
        _stats_T['freq'] = self.freq
        _stats = _stats_T.T

        # reorder
        _stats.index = pd.Categorical(_stats.index,
                                      ['returns',
                                       'volatility',
                                       'sharpe',
                                       'tstat',
                                       'maxDD',
                                       'start_dt',
                                       'end_dt',
                                       'freq',
                                       'order',
                                       'n_buckets'
                                       ])
        #import pdb; pdb.set_trace()

        #if self.n >= 10:
            #new_cols = [str(x) for x in np.arange(1,11)] + [_stats.columns[-1]]

            #_stats.columns = pd.Categorical(_stats.columns,
            #                          new_cols )
        _stats = _stats.reindex(columns=natsorted(_stats.columns))
        _stats = _stats.reindex(_stats.index.sort_values())

        return _stats

    @property
    def stats_one_liner(self):
        """
        
        Returns
        -------

        """
        _stats_brief = self.stats.iloc[0,:]
        _stats_brief['LS - sharpe'] = self.stats.iloc[:,-1].sharpe
        _stats_brief['LS - tstat'] = self.stats.iloc[:, -1].tstat
        _stats_brief['LS - maxDD'] = self.stats.iloc[:, -1].maxDD

        meta_stats = self.stats.T.loc['1', ['start_dt', 'end_dt', 'freq', 'order', 'n_buckets']]
        _stats_brief = pd.concat([_stats_brief,
                   meta_stats],
                  axis=0)

        return _stats_brief

    @property
    def portfolios(self):
        """

        Returns
        -------
        pd.DataFrame of portfolio mappings

        """
        return self.df_2.set_index([self.date_col_name,
                                    self.stock_col_name])[self.factor_col_q_name].unstack()

    @property
    def IC_time_series(self):
        """
        calculate time series of IC 

        Parameters
        ----------
        factor_name

        Returns
        -------
        dataframe
        """
        return self.data_df.groupby(self.date_col_name)[
            [self.factor_name, self.fwd_return_col_name]].apply(
            lambda x: x.corr('spearman').iloc[0, 1])

    @staticmethod
    def add_quintiles_as_new_col(df,
                                 col_name,
                                 new_col_name=None,
                                 groupby_col_name='date',
                                 n=10):
        """
        add quantiles as new columns to current df

        Parameters
        ----------
        df - pd.DataFrame
        col_name - columns name in which to quintile
        new_col_name - name for new column
        groupby_col_name - column to group by, default = 'date'
        n - number of buckets

        Returns
        -------
        pd.DataFrame

        """
        _df = df.copy()

        if new_col_name is None:
            new_col_name = '{}_q'.format(col_name)

        _df[new_col_name] = _df.groupby(groupby_col_name)[col_name].transform(lambda x: pd.qcut(
            x, q=n, labels=np.arange(1, n + 1))).astype(str).apply(lambda x: x.split('.')[0])
        _df[new_col_name].replace('nan', np.NaN, inplace=True)

        return _df

    def add_sector_neutral_column(self,
                                  df,
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

        #_df[neutralized_col_name] = _df.groupby(['date', 'sector'])[col_to_neutralize].rank(pct=True)
        _df[neutralized_col_name] = _df.groupby(
            agg_col_names)[col_to_neutralize].rank(pct=True)

        return _df


    @property
    def IC_avg(self):
        return self.IC_time_series.mean()

    def __repr__(self):
        return("""
SignalUnivariateStudy object v2.0
============================
fields = [stats, rets, wealth]
avg IC = {}

{}
        """.format(self.IC_avg,
                   self.stats))





