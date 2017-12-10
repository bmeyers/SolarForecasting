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
    * niter        - number of training iterations (default to 1000)
    * batchsize    - number of training examples in gradient approximation (default to 32)
    * learningrate - learning rate for gradient descent
    * sampling     - tells how to generate batches (rand or seq)
    * logdir       - directory for TensorBoard logs
    """
    def __init__(self, train, test, present=12*5, future=12*3,
                 train_selection="all", test_selection="hourly", logdir="./tmp/debug/trial",
                 arch="FC", learningrate=1e-2, niter=1000, batchsize=100, sampling="rand"):
        assert len(train) >= present + future, "present + future size must be smaller than training set"
        assert len(test) >= present, "present size must be smaller than test set"

        self.train = train
        self.test = test
        self.present = present
        self.future = future

        self.arch = arch
        self.niter = niter
        self.batchsize = batchsize
        self.learningrate = learningrate
        self.sampling = sampling

        self.features = train.iloc[:,0:-1] # assume inverters are in columns 2, 3, ..., n-1
        self.response = train.iloc[:,-1] # assume aggregate power is in column n

        self.features_dev = test.iloc[:,0:-1]
        self.response_dev = test.iloc[:,-1]

        self.ntrain = self.features.shape[0]
        self.ninverters = self.features.shape[1]

        self.ntest = self.features_dev.shape[0]

        if arch == "FC1":
            self.nn = FC([2000,1000,future], regularizer=l2(0.0001))

        self.logdir = logdir

    def inputdim(self):
        return self.present*self.ninverters

    def outputdim(self):
        return self.future

    def featurize(self, t, data="train"):
        '''
        Given a time stamp `t`, return the features and responses starting at `t`.
        '''
        if data == "train":
            features = self.features
            response = self.response
        elif data == "test":
            features = self.features_dev
            response = self.response_dev

        x = features[t:t+self.present].values.T.flatten().tolist()
        y = response[t+self.present:t+self.present+self.future].values.tolist()
        DoY = features.index.dayofyear[t]

        return x, y, DoY

    def make_batch(self, size, start, data="train"):
        """
        Produces batches of given `size` starting from time `start`.
        """

        X = []; Y = []
        D = []
        for i in range(size):
            t = start + i
            x, y, DoY = self.featurize(t, data=data)
            X.append(x)
            Y.append(y)
            D.append(DoY)

        # convert to Numpy
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        D = np.array(D, dtype=np.float32)
        D = D[:,np.newaxis]

        return X, Y, D

    def train_net(self, sess):
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

        # save nodes in the object for later use
        self.x = x
        self.y = y
        self.xdev = xdev
        self.ydev = ydev
        self.yhat = yhat
        self.yhatdev = yhatdev

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
        writer = tf.summary.FileWriter(self.logdir)
        writer.add_graph(sess.graph)

        tail = self.batchsize + self.present + self.future

        for i in range(self.niter):
            # define start of batch
            if self.sampling == "rand":
                start_train = np.random.randint(self.ntrain - tail)
            elif self.sampling == "seq":
                start_train = i % (self.ntrain - tail)

            start_dev = np.random.randint(self.ntest - tail)

            # create batch
            X, Y, D = self.make_batch(self.batchsize, start_train)
            Xdev, Ydev, Ddev = self.make_batch(self.batchsize, start_dev, data="test")

            if i % 5 == 0:
                [tloss, dloss, s] = sess.run([train_loss, dev_loss, summ],
                                             feed_dict={x: X, y: Y, day: D,
                                                        xdev: Xdev, ydev: Ydev, daydev: Ddev})
                writer.add_summary(s, i)
                writer.flush()

            sess.run(train_step, feed_dict={x: X, y: Y, day: D,
                                            xdev: Xdev, ydev: Ydev, daydev: Ddev})

    def make_forecasts(self):
        # Start clean
        tf.reset_default_graph()
        sess = tf.Session()

        # Fire up training, can take a while...
        self.train_net(sess)

        forecasts = []
        for t in np.arange(0, self.ntest - self.present - self.future + 1, 12):
            x, y, DoY = self.featurize(t, data="test")

            # reshape to row vector
            x = np.array(x, dtype=np.float32)[np.newaxis,:]

            # get result
            yhat = sess.run(self.yhatdev, feed_dict={self.xdev: x})

            # add back time information
            forecasts.append(pd.Series(data=yhat.flatten(), index=self.test.iloc[t+self.present:t+self.present+self.future].index))

        self.forecasts = forecasts
