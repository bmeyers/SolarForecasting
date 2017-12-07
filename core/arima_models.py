# -*- coding: utf-8 -*-
"""
This module contains ARIMA-based forecaster models
"""
from core.utilities import plot_forecasts, calc_test_mse
import numpy as np
import pandas as pd
from numpy.linalg import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SumToSumARIMA(object):
    def __init__(self, df):
        self.df = df
        self.train_start = None
        self.train_end = None
        self.order = None
        self.model = None
        self.model_fit = None
        self.forecasts = None
        self.test_set = None

    def train(self, start=None, end=None, order=(24, 0 , 1), seasonal_order=(0, 0, 0, 0), preset=None, method='lbfgs',
              maxiter=50):
        if preset == 'sunny':
            self.train_start = '2016-4-11'
            self.train_end = '2016-4-20'
        elif preset == 'cloudy':
            self.train_start = '2016-1-13'
            self.train_end = '2016-1-22'
        elif preset == 'mixed':
            self.train_start = '2015-10-9'
            self.train_end = '2015-10-18'
        else:
            self.train_start = start
            self.train_end = end
        self.order = order
        if start is None and end is None:
            train = self.df['total_power']
        else:
            train = self.df.loc[self.train_start:self.train_end]['total_power']
        model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                        enforce_invertibility=False, enforce_stationarity=False, maxiter=maxiter)
        fit = model.fit(method=method)
        self.model = model
        self.model_fit = fit
        return

    def test(self, test=None):
        if test is None:
            sunny_test = self.df.loc['2015-08-15']['total_power']
            cloudy_test = self.df.loc['2015-10-18']['total_power']
            test = pd.concat([sunny_test, cloudy_test], axis=0)
            start = test.index[0]
            ts = pd.date_range(start.date(), periods=len(test), freq='5min')
            test.index = ts
        elif isinstance(test, str):
            test = self.df.loc[test]['total_power']
        elif isinstance(test, tuple):
            test = self.df.loc[test[0]:test[1]]['total_power']
        forecasts = []
        N = len(test)/12
        for i in xrange(N - 1):
            next_batch = test.iloc[0:12 * (i + 1)]
            mod2 = SARIMAX(next_batch, order=self.order)
            test2 = mod2.filter(self.model_fit.params)
            forecast = test2.forecast(36)
            forecasts.append(forecast)
        self.forecasts = forecasts
        #self.forecasts.sort_index(inplace=True)
        self.test_set = test
        return

    def plot_test(self):
        plot_forecasts(self.test_set, self.forecasts)

    def calc_mse(self):
        mse = calc_test_mse(self.test_set, self.forecasts)
        return mse