# -*- coding: utf-8 -*-
"""
This module contains problem types that we are planning to address.
"""
from utilities import plot_forecasts, calc_test_mse
import numpy as np
import pandas as pd
from numpy.linalg import norm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

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
        plot_forecasts(self.test_set, self.forecasts)

    def calc_mse(self):
        mse = calc_test_mse(self.test_set, self.forecasts)
        return mse

class FunctionalRegression(object):
    def __init__(self, train, test, window=12*5, future=12*3, train_selection='all', test_selection='hourly'):
        if train_selection == 'all':
            self.train_features = np.array([train.iloc[i:i+window, :-1].values.ravel()
                                            for i in xrange(len(train) - window - future)])
            self.train_response = np.array([train.iloc[i+window:i+window+future, -1].values.ravel()
                                            for i in xrange(len(train) - window - future)])
        elif train_selection == 'hourly':
            self.train_features = np.array([train.iloc[i:i + window, :-1].values.ravel()
                                            for i in xrange(0, len(train) - window - future, 12)])
            self.train_response = np.array([train.iloc[i + window:i + window + future, -1].values.ravel()
                                            for i in xrange(0, len(train) - window - future, 12)])
        if test_selection == 'all':
            self.test_features = np.array([test.iloc[i:i+window, :-1].values.ravel()
                                            for i in xrange(len(test) - window - future)])
            self.test_response = np.array([test.iloc[i+window:i+window+future, -1].values.ravel()
                                            for i in xrange(len(test) - window - future)])
            self.test_response_ts = np.array([test.iloc[i+window:i+window+future, -1].index
                                            for i in xrange(len(test) - window - future)])
        elif test_selection == 'hourly':
            self.test_features = np.array([test.iloc[i:i + window, :-1].values.ravel()
                                            for i in xrange(0, len(test) - window - future, 12)])
            self.test_response = np.array([test.iloc[i + window:i + window + future, -1].values.ravel()
                                            for i in xrange(0, len(test) - window - future, 12)])
            self.test_response_ts = np.array([test.iloc[i + window:i + window + future, -1].index
                                           for i in xrange(0, len(test) - window - future, 12)])
        self.train = train
        self.test = test
        self.window = window
        self.future = future
        self.distances = None
        self. neighborhoods = None
        self.forecasts = None

    def make_forecasts(self, neighborhood=5):
        self.calc_distances_and_neighborhoods(neighborhood=neighborhood)
        p, n = self.test_features.shape
        forecasts = []
        forecasts_a = forecasts.append
        for j in xrange(p):
            neighb = self.neighborhoods[j]
            ds = self.distances[j, neighb]
            h = np.max(self.distances[j, :])
            num = 0
            den = 0
            for i in neighb:
                k = self.ker(self.distances[j, i] / h)
                num += k * self.train_response[i, :]
                den += k
            est = pd.Series(data=(num / den), index=self.test_response_ts[j])
            forecasts_a(est)
        # Cast the forecasts as Pandas Series with time indices
        foo = [pd.Series(data=f, index=self.test_response_ts[ind]) for ind, f in enumerate(forecasts)]
        self.forecasts = foo

    def calc_distance(self, feature1, feature2):
        residual = feature1 - feature2
        distance = np.sqrt(residual.dot(residual))
        return distance

    def calc_distances_and_neighborhoods(self, neighborhood=5):
        m, n = self.train_features.shape
        p, n = self.test_features.shape
        d = [[self.calc_distance(self.test_features[j, :], self.train_features[i, :])
              for i in xrange(m)] for j in xrange(p)]
        n = [np.argsort(d[i])[:neighborhood] for i in xrange(p)]
        self.distances = np.array(d)
        self.neighborhoods = np.array(n)

    def ker(self, t):
        output = np.max(np.c_[1 - t, np.zeros_like(t)], axis=1)
        if len(output) == 1:
            return output[0]
        else:
            return output

    def plot_test(self, start=None, end=None):
        plot_forecasts(self.test, self.forecasts)


    def calc_mse(self):
        mse = calc_test_mse(self.test, self.forecasts)
        return mse


if __name__ == "__main__":
    df = pd.read_pickle('/Users/bennetmeyers/Documents/CS229/Project/SolarForecasting/data/master_dataset.pkl')
    df.fillna(0, inplace=True)
    prob = SumToSumARIMA(df)
    prob.train(preset='sunny')
    prob.test()