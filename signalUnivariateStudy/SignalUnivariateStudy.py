

import numpy as np
import pandas as pd

from signalUnivariateStudy.functions import calc_stats, add_quintiles_as_new_col, add_sector_neutral_column


class Factor(object):
    """
    
    """
    pass
    #raise NotImplementedError()


class SignalUnivariateStudy(object):
    """
    
    """

    def __init__(self,
                 data_df,
                 factor_name,
                 neutralizer_column = None,
                 order = 'asc',
                 n = 5,
                 date_col_name = 'date',
                 fwd_return_col_name = 'fwd_returns',
                 start_dt = None,
                 end_dt = None,
                 sector = None):
        """
        
        Parameters
        ----------
        data_df
        factor_name
        order
        n
        date_col_name
        fwd_return_col_name
        start_dt
        end_dt
        sector
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
        self.stats = None
        self.neutralizer_column = neutralizer_column

        # fix fwd returns
        if self.data_df[fwd_return_col_name].mean() > 0.99:
            self.data_df[fwd_return_col_name] = self.data_df[fwd_return_col_name] * 0.01

        self.run_backtest()


    def run_backtest(self):
        """
        
        Returns
        -------

        """
        #_df = self.data_df.copy()

        if self.start_dt is not None:
            _start = self.start_dt
            print("starting backtest on {}".format(_start))
            self.data_df = self.data_df.query("date >= @_start ")

        if self.end_dt is not None:
            _end = self.end_dt
            print("ending backtest on {}".format(_end))
            self.data_df = self.data_df.query("date <= @_end ")
        if self.neutralizer_column is not None:
            print("neutralizing factor = {} using {}".format(self.factor_name,
                                                             self.neutralizer_column))
            self.data_df = add_sector_neutral_column(df=self.data_df,
                                       col_to_neutralize= self.factor_name,
                                       neutralized_col_name= None,
                                       agg_col_names=['date', self.neutralizer_column])


            df_2 = add_quintiles_as_new_col(df = self.data_df,
                                            col_name= '{}_SN'.format(self.factor_name),
                                            new_col_name=None,
                                            groupby_col_name=self.date_col_name,
                                            n=self.n)
            self.factor_col_q_name = '{}_SN_q'.format(self.factor_name)
        else:
            df_2 = add_quintiles_as_new_col(df=self.data_df,
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
        self.stats = pd.DataFrame(self.rets.apply(lambda x: pd.Series(calc_stats(x))))

        self.wealth = np.cumprod(1 + self.rets)

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

    @property
    def IC_avg(self):
        return self.IC_time_series.mean()

    def __repr__(self):
        return("""
SignalUnivariateStudy object
============================
fields = [stats, rets, wealth]
avg IC = {}

{}
        """.format(self.IC_avg,
                   self.stats))





