# -*- coding: utf-8 -*-
"""
This module contains problem types that we are planning to address.
"""
from core.utilities import plot_forecasts, calc_test_mse
import numpy as np
import pandas as pd

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range



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
