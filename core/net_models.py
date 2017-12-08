import tensorflow as tf
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from keras.regularizers import l1, l2


class FC(object):
    """
    Fully-connected network
    """
    def __init__(self, layer_units):
        nlayers = len(layer_units)
        self.layers = []
        for i in range(nlayers-1):
            self.layers.append(Dense(layer_units[i], activation=tf.nn.relu))
        self.layers.append(Dense(layer_units[-1]))

    def __call__(self, x):
        a = x
        for l in self.layers:
            a = l(a)
        return a


class CNN(object):
    """
    Convolutional network

    Given a flattened input example x, reshape it to an image with
    height and width and feed forward in the net.
    """
    def __init__(self, layer_units, height, width):
        nlayers = len(layer_units)
        self.height = height
        self.width = width
        self.layers = []
        for i in range(nlayers-1):
            self.layers.append(Conv2D(layer_units[i], kernel_size=5, activation=tf.nn.relu, padding='same'))
            self.layers.append(MaxPool2D(padding='same'))
        self.layers.append(Flatten())
        self.layers.append(Dense(layer_units[-1]))

    def __call__(self, x):
        a = tf.reshape(x, [-1, self.height, self.width, 1])
        for l in self.layers:
            a = l(a)
        return a
