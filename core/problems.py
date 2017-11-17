# -*- coding: utf-8 -*-
"""
This module contains problem types that we are planning to address.
"""
import numpy as np

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
