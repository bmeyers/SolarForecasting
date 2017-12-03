# -*- coding: utf-8 -*-

from core.forecaster import Forecaster

import numpy as np
import pandas as pd
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

    # Additional constructor arguments:

    * arch      - neural network architecture (default to "dense")
    * nepochs   - number of training epochs (default to 100)
    * trainsize - size of training set for each epoch (default to len(train))
    * batchsize - number of training examples in gradient approximation (default to 32)
    '''
    def __init__(self, train, test, window=12*5, future=12*3,
                 train_selection="all", test_selection="hourly",
                 arch="dense", nepochs=100, trainsize=None, batchsize=32):
        assert len(train) >= window + future, "window + future size must be smaller than training set"
        assert len(test) >= window, "window size must be smaller than test set"

        self.train = train
        self.test = test
        self.window = window
        self.future = future

        self.arch = arch
        self.nepochs = nepochs
        self.trainsize = trainsize
        self.batchsize = batchsize

        self.features = train.iloc[:,1:-1] # assume inverters are in columns 2, 3, ..., n-1
        self.response = train.iloc[:,-1] # assume aggregate power is in column n

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

    def sampler(self, trainsize=None):
        '''
        This generator takes the dataset and produces a batch of training
        examples at random.
        '''
        nstamps, ninverters = self.features.shape

        # create a batch of examples of the same size of the
        # dataframe that we already know fits in memory
        if trainsize is None:
            trainsize = nstamps

        while True:
            X = []; Y = []
            for _ in range(trainsize):
                t = np.random.randint(nstamps-self.window-self.future+1)

                x, y = self.featurize(t)

                X.append(x)
                Y.append(y)

            yield X, Y

    def make_forecasts(self):
        if self.arch == "dense": # FULLY CONNECTED
            model = Sequential([
                Dense(50, activation='relu', input_shape=(self.inputdim(),)),
                Dense(50, activation='relu', kernel_regularizer=l2(.1)),
                Dense(50, activation='relu', kernel_regularizer=l2(.1)),
                Dense(self.outputdim(), activation='linear')
            ])
        elif self.arch == "conv": # CONVOLUTIONAL
            model = Sequential([
                Conv1D(25, kernel_size=30, input_shape=(self.inputdim(),1)),
                MaxPool1D(),
                Conv1D(10, kernel_size=15),
                Flatten(),
                Dense(self.outputdim(), activation='linear')
            ])
        elif self.arch == "lstm": # RECURRENT
            model = Sequential([
                LSTM(25, activation='relu', input_shape=(self.inputdim(),1)),
                Dense(self.outputdim(), activation='linear')
            ])
        else:
            raise ValueError("invalid neural network architecture")

        print(model.summary())

        model.compile(loss='mean_squared_error', optimizer=Adamax(), metrics=['mean_squared_error'])

        # after hours trying to debug Keras in order to use model.fit_generator(),
        # I decided to just do it manually for now:
        sampler = self.sampler(trainsize=self.trainsize)
        for i in range(self.nepochs):
            print("Iteration", i)

            X, Y = next(sampler)

            # handle Keras weird conventions
            if self.arch == 'conv' or self.arch == 'lstm':
                X = np.reshape(X, (len(X),len(X[0]),1))

            model.fit(X, Y, batch_size=self.batchsize, epochs=1, validation_split=.1, verbose=1)

        forecasts = []
        for t in np.arange(0, len(self.test) - self.window - self.future + 1, 12):
            x, y = self.featurize(t)

            if self.arch == 'conv' or self.arch == 'lstm':
                x = np.reshape(x, (len(x), 1))

            yhat = model.predict(np.array([x]))

            forecasts.append(pd.Series(data=yhat.flatten(), index=self.test.iloc[t+self.window:t+self.window+self.future].index))

        self.forecasts = forecasts
