# -*- coding: utf-8 -*-

from shutil import rmtree
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.regularizers import l1, l2

from core.forecaster import Forecaster
from core.net_models import FC, CNN
from core.preprocessing import center_design_matrix

# Python 2.x, 3.x compatibility
try:
    xrange
except NameError:
    xrange = range


class NeuralNetForecaster(Forecaster):
    """
    Many-to-one neural network regression in which past time series
    from many inverters are used to predict their aggregate power in
    the future.

    # Additional constructor arguments:

    * arch         - neural network architecture (default to "dense")
    * nepochs      - number of training epochs (default to 100)
    * batchsize    - number of training examples in gradient approximation (default to 32)
    * learningrate - learning rate for gradient descent
    * logdir       - directory for TensorBoard logs
    * rmlogdir     - tells whether or not to remove an existing TensorBoard directory log
    """
    def __init__(self, dftrain, dftest, present=12*5, future=12*3,
                 train_selection="all", test_selection="hourly", logdir="./tmp/debug/", rmlogdir=False,
                 arch="FC", learningrate=1e-2, nepochs=1000, batchsize=100):
        assert len(dftrain) >= present + future, "present + future size must be smaller than training set"
        assert len(dftest) >= present, "present size must be smaller than test set"

        self.dftrain = dftrain
        self.dftest = dftest
        self.present = present
        self.future = future

        self.arch = arch
        self.nepochs = nepochs
        self.batchsize = batchsize
        self.learningrate = learningrate

        self.features = dftrain.iloc[:,0:-1] # assume inverters are in columns 2, 3, ..., n-1
        self.response = dftrain.iloc[:,-1] # assume aggregate power is in column n

        self.nstamps = self.features.shape[0]
        self.ninverters = self.features.shape[1]

        if arch == "FC":
            self.nn = FC([512,512,future])
        elif arch == "CNN":
            self.nn = None

        self.logdir = logdir

        if rmlogdir:
            rmtree(logdir + arch)

    def inputdim(self):
        return self.present*self.ninverters

    def outputdim(self):
        return self.future

    def featurize(self, t):
        '''
        Given a time stamp `t`, return the features and responses starting at `t`.
        '''
        x = self.features[t:t+self.present].values.T.flatten().tolist()
        y = self.response[t+self.present:t+self.present+self.future].values.tolist()
        DoY = self.features.index.dayofyear[t]

        return x, y, DoY

    def make_batch(self, size):
        """
        Produces batches of given size.
        """

        X = []; Y = []
        D = []
        for i in range(size):
            t = np.random.randint(self.nstamps - self.present - self.future)
            x, y, DoY = self.featurize(t)
            X.append(x)
            Y.append(y)
            D.append(DoY)

        # convert to Numpy
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        D = np.array(D, dtype=np.float32)
        D = D[:,np.newaxis]

        return X, Y, D

    def train(self, sess):
        # Setup placeholders for input and output
        x = tf.placeholder(tf.float32, shape=[None, self.present*self.ninverters], name="x")
        y = tf.placeholder(tf.float32, shape=[None, self.future], name="y")

        # Similarly, setup placeholders for dev set
        xdev = tf.placeholder(tf.float32, shape=[None, self.present*self.ninverters], name="xdev")
        ydev = tf.placeholder(tf.float32, shape=[None, self.future], name="ydev")

        # Setup placeholders for day of year
        day = tf.placeholder(tf.float32, shape=[None, 1], name="day")
        daydev = tf.placeholder(tf.float32, shape=[None, 1], name="daydev")

        # Feed forward into the net
        yhat    = self.nn(x)
        yhatdev = self.nn(xdev)

        # Define loss in training and dev set
        with tf.name_scope("loss"):
            train_loss = tf.losses.mean_squared_error(labels=y, predictions=yhat)
            dev_loss = tf.losses.mean_squared_error(labels=ydev, predictions=yhatdev)
            tf.summary.scalar("train_loss", train_loss)
            tf.summary.scalar("dev_loss", dev_loss)

        # Minimize training loss
        with tf.name_scope("train"):
            train_step = tf.train.AdamOptimizer(self.learningrate).minimize(train_loss)

        # Collect all summaries for TensorBoard
        summ = tf.summary.merge_all()

        # Start of execution
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter(self.logdir + self.arch)
        writer.add_graph(sess.graph)

        for i in range(self.nepochs):
            # create batch
            X, Y, D = self.make_batch(self.batchsize)
            Xdev, Ydev, Ddev = self.make_batch(self.batchsize)

            # center data
            X = center_design_matrix(X)
            Xdev = center_design_matrix(Xdev)

            if i % 5 == 0:
                [tloss, dloss, s] = sess.run([train_loss, dev_loss, summ],
                                             feed_dict={x: X, y: Y, day: D,
                                                        xdev: Xdev, ydev: Ydev, daydev: Ddev})
                writer.add_summary(s, i)
                writer.flush()

            if i % 100 == 0:
                print("Done with batch {}".format(i+1))

            sess.run(train_step, feed_dict={x: X, y: Y, day: D,
                                            xdev: Xdev, ydev: Ydev, daydev: Ddev})

    def make_forecasts(self):
        # Start clean
        tf.reset_default_graph()
        sess = tf.Session()

        # Fire up training, can take a while...
        self.train(sess)

        forecasts = []
        for t in np.arange(0, len(self.dftest) - self.present - self.future + 1, 12):
            x, y, DoY = self.featurize(t)

            # TODO: call self.nn(x) and add time index
            # yhat = self.nn(x)
            # forecasts.append(pd.Series(data=yhat.flatten(), index=self.dftest.iloc[t+self.present:t+self.present+self.future].index))

        self.forecasts = forecasts
