# -*- coding: utf-8 -*-
"""
This module contains utility functions and classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cvx

def plot_forecasts(test, forecasts, ax=None):
    '''
    A utility function for plotting a list of forecasts over the target data
    :param test: the test dataframe, with a time index and 'total_power' column
    :param forecasts: a list of forecasts, formatted as Pandas series with time indices
    :param ax: optional, an axis object
    :return: a matplotlib figure unless an axis object is passed
    '''
    if ax is None:
        fig, ax = plt.subplots()
    test.plot(y='total_power', linewidth=1.5, ls=':', ax=ax)
    for series in forecasts:
        series.plot(linewidth=1, ax=ax)
    if ax is None:
        return fig


def calc_test_mse(test, forecasts):
    '''
    A utility function for calculating the mean square error of a batch of forecasts from
    the target data. Only forecasts periods with at least one non-zero piece of data are
    considered.
    :param test: the test dataframe, with a time index and 'total_power' column
    :param forecasts: a list of forecasts, formatted as Pandas series with time indices
    :return: the mean square error over all forecasts
    '''
    residuals = np.array([])
    for f in forecasts:
        join = pd.concat([f, test], axis=1).dropna()
        if np.max(join['total_power']) > 0:
            r = join['total_power'] - join[0]
            residuals = np.r_[residuals, r]
    return np.sum(np.power(residuals, 2)) / np.float(len(residuals))

def envelope_fit(signal, mu, eta, kind='upper', period=None):
    '''
    Perform an envelope fit of a signal. See: https://en.wikipedia.org/wiki/Envelope_(waves)
    :param signal: The signal to be fit
    :param mu: A parameter to control the overall stiffness of the fit
    :param eta: A parameter to control the permeability of the envelope. A large value result in
    no data points outside the fitted envelope
    :param kind: 'upper' or 'lower'
    :return: An envelope signal of the same length as the input
    '''
    if kind == 'lower':
        signal *= -1
    n_samples = len(signal)
    envelope = cvx.Variable(len(signal))
    mu = cvx.Parameter(sign='positive', value=mu)
    eta = cvx.Parameter(sign='positive', value=eta)
    cost = (cvx.norm2(envelope - signal) +
            mu * cvx.norm2(envelope[2:] - 2 * envelope[1:-1] + envelope[:-2]) +
            eta * cvx.norm1(cvx.max_elemwise(signal - envelope, 0)))
    objective = cvx.Minimize(cost)
    if period is not None:
        constraints = [
            envelope[:n_samples - period] == envelope[period:]
        ]
    problem = cvx.Problem(objective, constraints)
    problem.solve(solver='ECOS')
    if kind == 'upper':
        return envelope.value.A1
    elif kind == 'lower':
        signal *= -1
        return -envelope.value.A1