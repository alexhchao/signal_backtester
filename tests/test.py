

import unittest

import pandas as pd

from adaboost.adaBoostModelClass import adaBoostModel

import pandas as pd
import numpy as np
import os
import sys
import seaborn as sns
sns.set()
from signalUnivariateStudy.SignalUnivariateStudy import SignalUnivariateStudy


class TestSum(unittest.TestCase):

    def test_toy_example(self):
        list_factors = ['sector', 'momentum', 'quality', 'growth', 'vol', 'value', 'size']

        df = pd.read_csv('../data/stock_data_actual_dates.csv').iloc[:, 1:]

        sig = SignalUnivariateStudy(data_df=df,
                                    factor_name='momentum',
                                    stock_col_name = 'stock',
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
        #self.assertEqual(1, 2, "predictions dont match...")

    #def test_sum_tuple(self):
    #    self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")

if __name__ == '__main__':
    unittest.main()


