# -*- coding: utf-8 -*-

from core.forecaster import Forecaster

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPool1D, MaxPool2D, Flatten
from keras.regularizers import l1, l2
from keras.optimizers import Adamax, SGD

class NeuralNetForecaster(Forecaster):
    """
    Many-to-one neural network regression in which past time series
    from many inverters are used to predict their aggregate power in
    the future.

    # Additional constructor arguments:

    * arch      - neural network architecture (default to "dense")
    * nepochs   - number of training epochs (default to 100)
    * trainsize - size of training set for each epoch (default to len(train))
    * batchsize - number of training examples in gradient approximation (default to 32)
    """
    def __init__(self, train, test, window=12*5, future=12*3,
                 train_selection="all", test_selection="hourly",
                 arch="dense", nepochs=50, trainsize=None, batchsize=32):
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

        self.features = train.iloc[:,0:-1] # assume inverters are in columns 2, 3, ..., n-1
        self.response = train.iloc[:,-1] # assume aggregate power is in column n

    def featurize(self, t):
        '''
        Given a time stamp `t`, return the features and responses starting at `t`.
        '''
        x = self.features[t:t+self.window].values.T.flatten().tolist()
        y = self.response[t+self.window:t+self.window+self.future].values.tolist()

        return x, y

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

            yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

    def make_forecasts(self):
        if self.arch == "dense": # FULLY CONNECTED
            model = Sequential([
                Dense(1024, activation='relu', input_shape=(self.inputdim(),)),
                Dense(512, activation='relu'),
                Dense(50, activation='relu'),
                Dense(self.outputdim(), activation='linear')
            ])
        elif self.arch == "conv": # CONVOLUTIONAL
            model = Sequential([
                Conv1D(64, kernel_size=3, input_shape=(self.inputdim(),1)),
                MaxPool1D(),
                Conv1D(32, kernel_size=3),
                MaxPool1D(),
                Flatten(),
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
