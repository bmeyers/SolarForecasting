# -*- coding: utf-8 -*-

from core.forecaster import Forecaster

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, LSTM, Dropout, MaxPool1D, Flatten
from keras.regularizers import l1, l2
from keras.optimizers import Adamax, SGD

class NeuralNetForecaster(Forecaster):
    '''
    Many-to-one neural network regression in which past time series
    from many inverters are used to predict their aggregate power in
    the future.
    '''
    def __init__(self, train, test, window=12*5, future=12*3,
                 train_selection="all", test_selection="hourly"):
        assert len(train) >= window + future, "window + future size must be smaller than training set"
        assert len(test) >= window, "window size must be smaller than test set"

        self.train = train
        self.test = test
        self.window = window
        self.future = future

        self.features = train.iloc[:,1:-1]
        self.response = train.iloc[:,-1]

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

    def sampler(self, batchsize=None):
        '''
        This generator takes the dataset and produces a batch of training
        examples at random.
        '''
        nstamps, ninverters = self.features.shape

        # create a batch of examples of the same size of the
        # dataframe that we already know fits in memory
        if batchsize is None:
            batchsize = nstamps

        while True:
            X = []; Y = []
            for _ in range(batchsize):
                t = np.random.randint(nstamps-self.window-self.future+1)

                x, y = self.featurize(t)

                X.append(x)
                Y.append(y)

            yield X, Y

    def make_forecasts(self):
        pass # TODO
