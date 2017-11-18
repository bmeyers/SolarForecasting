# -*- coding: utf-8 -*-
"""
This module contains problem types that we are planning to address.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

class ManyToOneRegression(object):
    '''
    This class represents the many-to-one problem where we use past series
    from many inverters to predict their aggregate in the future.
    '''
    def __init__(self, df, window=256, future=50):
        self.features = df.iloc[:,1:-1]
        self.response = df.iloc[:,-1]
        self.window = window
        self.future = future

    def featurize(self, t):
        '''
        Given a time stamp `t`, return the features and responses starting at `t`.
        '''
        x = self.features[t:t+self.window].values.T.flatten().tolist()
        y = self.response[t+self.window:t+self.window+self.future].values.tolist()

        return x, y

    def unfeaturize(self, t, x, y):
        '''
        Given a time stamp `t`, and transformed features `x` and responses `y`, undo
        the featurization, that is, convert `x` back into a "matrix format" with columns
        for each inverter. `y` is not touched.
        '''
        m, n = self.features.shape
        nsteps = len(x) // n

        series = np.array(x).reshape((nsteps,n))
        aggregate = np.array(y)

        return series, aggregate

    def inputdim(self):
        nstamps, ninverters = self.features.shape
        return self.window*ninverters

    def outputdim(self):
        return self.future

    def sampler(self, batchsize=32):
        '''
        This generator takes the dataset and produces a batch of training
        examples at random.
        '''
        nstamps, ninverters = self.features.shape

        while True:
            X = []; Y = []
            for _ in range(batchsize):
                t = np.random.randint(nstamps-self.window-self.future+1)

                x, y = self.featurize(t)

                X.append(x)
                Y.append(y)

            yield X, Y


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
        self.test_set.plot(linewidth=1.5, ls=':')
        for f in self.forecasts:
            f.plot(linewidth=1)

    def calc_mse(self):
        residuals = np.array([])
        for f in self.forecasts:
            join = pd.concat([f, self.test_set], axis=1).dropna()
            if np.max(join['total_power']) > 0:
                r = join['total_power'] - join[0]
                residuals = np.r_[residuals, r]
        return np.sum(np.power(residuals, 2)) / np.float(len(residuals))

class FunctionalRegression(object):
    def __init__(self, df):
        pass

if __name__ == "__main__":
    df = pd.read_pickle('/Users/bennetmeyers/Documents/CS229/Project/SolarForecasting/data/master_dataset.pkl')
    df.fillna(0, inplace=True)
    prob = SumToSumARIMA(df)
    prob.train(preset='sunny')
    prob.test()