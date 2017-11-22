# -*- coding: utf-8 -*-
"""
This module contains classes and functions for pre-processing data and initial data selection.
"""

from core.utilities import envelope_fit
import numpy as np
from numpy.linalg import svd
import cvxpy as cvx
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

TRAIN = ('2015-7-15', '2016-11-25') # 500 days
DEV = ('2016-11-26', '2017-3-15')   # 110 days
TEST = ('2017-3-16', '2017-07-14')  # 121 days

class StatisticalClearSky(object):
    def __init__(self, data):
        self.data = data
        self.U = None
        self.D = None
        self.P = None
        self.DP_clearsky = None

    def get_eigenvectors(self):
        data_matrix = self.data.as_matrix().reshape(-1, 288).T
        U, D, P = svd(data_matrix)
        self.U = U
        self.D = D
        self.P = P
        self.data = data_matrix

    def reconstruct_day(self, day=20, n=100, plot=True):
        if self.U is None:
            self.get_eigenvectors()
        if plot:
            plt.plot(self.data[:, day], linewidth=1)
            plt.plot(self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day]), linewidth=1)
        else:
            return self.data[:, day], self.U[:, :n].dot(np.diag(self.D[:n])).dot(self.P[:n, day])

    def make_clearsky_model(self, n=5, mu=10**(2.), eta=10**(-1.), plot=False):
        if self.U is None:
            self.get_eigenvectors()
        daily_scale_factors = ((np.diag(self.D).dot(self.P[:288])))
        signals = []
        fits = np.empty((n, self.P.shape[1]))
        for ind in xrange(n):
            signal = daily_scale_factors[ind, :]
            if ind == 0:
                envelope = envelope_fit(signal, mu=mu, eta=eta, kind='lower', period=365)
            elif ind == 1:
                envelope = envelope_fit(signal, mu=mu, eta=10**(-10), kind='upper', period=365)
            else:
                envelope = envelope_fit(signal, mu=mu, eta=10**(-10), kind='upper', period=365)
            signals.append(signal)
            fits[ind, :] = envelope
        self.DP_clearsky = fits[:, :365]
        if plot:
            fig, axes = plt.subplots(nrows=n, figsize=(12,14))
            for ind in xrange(n):
                axes[ind].plot(signals[ind], linewidth=1)
                axes[ind].plot(fits[ind], linewidth=1)
            return fig, axes

    def estimate_clearsky(self, day_slice):
        '''
        Make a clearsky estimate based on provided data for a given set of days
        :param day_slice: A numpy.slice object indicating the days to be used in the estimation (see: numpy.s_)
        :return: A matrix of daily clear sky estimates. The columns are individual days and the rows 5-minute
        time periods
        '''
        if self.DP_clearsky is None:
            self.make_clearsky_model()
        n = self.DP_clearsky.shape[0]
        clearsky = self.U[:, :n].dot(self.DP_clearsky[:, day_slice])
        return clearsky.clip(min=0)


def data_summary(df):
    '''
    This function accepts a Pandas dataframe generated from a SunPower type-130 data file.
    :param df: Pandas dataframe
    :return: a dictionary containing the starting timestamp, end timestamp, number of non-null data
             points, and the max, min, average, and standard deviation of the AC power column for every inverter serial
             number in the file
    '''
    df.sort_values('timestamp', inplace=True)
    serial_numbers = set(df['serial_number'])
    output = {}
    for sn in serial_numbers:
        df_view = df[df['serial_number'] == sn]
        df_view = df_view[pd.notnull(df_view['ac_power'])]
        if len(df_view) != 0:
            tstart = df['timestamp'].iloc[0]
            tend = df['timestamp'].iloc[-1]
            nvals = len(df_view)
            ac_max = np.max(df_view['ac_power'])
            ac_min = np.min(df_view['ac_power'])
            ac_avg = np.mean(df_view['ac_power'])
            ac_stdev = np.std(df_view['ac_power'])
            output[sn] = {
                't_start': tstart,
                't_end': tend,
                'num_vals': nvals,
                'ac_max': ac_max,
                'ac_min': ac_min,
                'ac_avg': ac_avg,
                'ac_stdev': ac_stdev
            }
    return output

def load_raw_file(filename, kind='csv'):
    '''
    Because of explicit dtype handling, this will only work on type-130 raw SunPower data files. Handles CSVs and
    compressed CSVs, including .gz and .zip
    :param filename: the file path to a SunPower type-130 raw data file
    :return: a pandas
    '''
    if kind in ['gz', 'csv', 'zip']:
        dtypes_1 = {
            'key': np.int,
            'serial_number': str,
            'inverter_serial_number': str,
            'hardware_version': str,
            'software_version': np.float,
            'inverter_model_number': str,
            'model_d1': str,
            'model_d2': np.float,
            'energy_harvested_today': np.float,
            'energy_harvested_total': np.float,
            'dc_voltage': np.float,
            'dc_current': np.float,
            'dc_power': np.float,
            'ac_voltage': np.float,
            'ac_current': np.float,
            'ac_power': np.float,
            'ac_frequency': np.float,
            'heatsink_temp': np.float
        }
        dtypes_2 = {
            'key': np.int,
            'serial_number': str,
            'model_name': str,
            'energy_day': np.float,
            'energy_total': np.float,
            'ac_power': np.float,
            'ac_volt': np.float,
            'ac_curr': np.float,
            'dc_power': np.float,
            'dc_volt': np.float,
            'dc_curr': np.float,
            'heatsink_temp': np.float,
            'freq': np.float,
            'inverter_status': int
        }
        df = pd.read_csv(filename, index_col=False, parse_dates=[1], na_values=['XXXXXXX'])
        try:
            df.astype(dtypes_1)
        except KeyError:
            df.astype(dtypes_2)
    elif kind == 'pkl':
        df = pd.read_pickle(filename)
    return df

def summarize_files(file_path, suffix='gz', verbose=False, testing=False):
    '''
    Provide a high-level summary of all files in directory. Calls data_summary()
    :param file_path: a string containing the path to the directory containing the files to be summarized
    :param suffix: options are 'gz', 'zip', 'csv', or 'pkl'
    ;param verbose: boolean to print progress
    :return: a pandas dataframe with the summary of the files
    '''
    if not file_path[-1] == '/':
        file_path += '/'
    search = file_path + '*.' + suffix
    files = glob(search)
    if testing:
        files = files[:40]
    data = {}
    N = len(files)
    print('{} files to process'.format(N))
    for it, fn in enumerate(files):
        df = load_raw_file(fn, kind=suffix)
        summary = data_summary(df)
        name = fn.split('/')[-1]
        name = name.split('.')[0]
        split_name = name.split('_')
        key_base = split_name[0] + '_' + split_name[2]
        for key, val in summary.iteritems():
            full_key = key_base + '_' + str(key)
            data[full_key] = val
        if verbose:
            print('{}/{} complete:'.format(it+1, N), name)
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df[['t_start', 't_end', 'num_vals', 'ac_max', 'ac_min', 'ac_avg', 'ac_stdev']]
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'system key'}, inplace=True)
    sites = df['system key'].str.split('_').str.get(0)
    df['site ID'] = sites
    return df

def pickle_files(file_path, suffix='gz', verbose=False):
    '''
    Parse files with pandas and save as pickles for faster access in the future
    :param file_path: a string containing the path to the directory containing the files to be summarized
    :param suffix: options are 'gz', 'zip', or 'csv'
    ;param verbose: boolean to print progress
    :return:
    '''
    if not file_path[-1] == '/':
        file_path += '/'
    search = file_path + '*.' + suffix
    files = glob(search)
    N = len(files)
    if verbose:
        print('{} files to process'.format(N))
    for it, fn in enumerate(files):
        df = load_raw_file(fn)
        name = fn.split('/')[-1]
        name = name.split('.')[0]
        save_path = file_path + name + '.pkl'
        df.to_pickle(save_path)
        if verbose:
            print('{}/{} complete:'.format(it+1, N), name)

def generate_master_dataset(site_ids, file_path, suffix='pkl', verbose=False):
    '''
    Parse the files associated with a list of site ids, combining AC power from all inverters at each site into a single
    data frame with a regular time-series index.
    :param site_keys: either an interable of site keys or a file path to a text file with the list
    :param file_path: file path to the raw data files
    :param suffix: 'pkl' is fastest. use 'csv' for original files
    :param verbose: boolean to print progress
    :return: the processed dataframe and a list of site key, name pairs
    '''
    # duck typing for file path or iterable
    if isinstance(site_ids, str):
        site_ids = np.genfromtxt(site_ids, dtype=str)
    # set the master time index that all data will match to
    time_index = pd.date_range(start='2015-07-15', end='2017-07-16', freq='5min')
    site_keys = []
    site_keys_a = site_keys.append
    output = pd.DataFrame(index=time_index)
    counter = 1
    N = len(site_ids)
    if verbose:
        print('{} files to process'.format(N))
    for it, id in enumerate(site_ids):
        fp = glob(file_path + '*' + id +'*' + suffix)[0]
        df = load_raw_file(fp, kind=suffix)
        serial_numbers = set(df['serial_number'])
        # some sites have more than one inverter
        for sn in serial_numbers:
            df_view = df[df['serial_number'] == sn]
            ############## data cleaning ####################################
            df_view = df_view[pd.notnull(df_view['ac_power'])]              # Drop records with nulls in ac_power
            df_view.set_index('timestamp', inplace=True)                    # Make the timestamp column the index
            df_view.sort_index(inplace=True)                                # Sort on time
            df_view = df_view[~df_view.index.duplicated(keep='first')]      # Drop duplicate times
            df_view.reindex(index=time_index).interpolate()                 # Match the master index, interp missing
            #################################################################
            col_id = str(id) + '_' + str(sn)
            col_name = 'S{:02}'.format(counter)
            output[col_name] = df_view['ac_power']
            if output[col_name].count() > 200: # final filter on low data count relative to time index
                site_keys_a((col_id, col_name))
                counter += 1
            else:
                del output[col_name]
        if verbose:
            print('{}/{} complete:'.format(it+1, N), id)
    total_power = np.sum(output, axis=1)
    output['total_power'] = total_power
    output.index = output.index + pd.Timedelta(hours=-8) # Localize time to UTC-8
    return output, site_keys

def train_dev_test_split(df, train=TRAIN, dev=DEV, test=TEST):
    '''
    A helper function for doing train, dev, test splitting. The standard date ranges are the defaults. Set a kwarg to
    None to suppress the return of that subset. For instance, if you are doing model training, you probably don't need
    the test set until the very end.
    :param df: A dataframe. Should probably be the master dataset and nothing else
    :param train: a tuple containing the start and end days of the train period
    :param dev: a tuple containing the start and end days of the dev period
    :param test: a tuple containing the start and end days of the test period
    :return: a list of split dataframes
    '''
    if train is not None:
        df_train = df.loc[train[0]:train[1]]
    else:
        df_train = None
    if dev is not None:
        df_dev = df.loc[dev[0]:dev[1]]
    else:
        df_dev = None
    if test is not None:
        df_test = df.loc[test[0]:test[1]]
    else:
        df_test = None
    return [item for item in [df_train, df_dev, df_test] if item is not None]

def make_small_train(df, kind='mixed'):
    if kind == 'sunny':
        start = '2016-4-11'
        end = '2016-4-20'
    elif kind == 'cloudy':
        start = '2016-1-13'
        end = '2016-1-22'
    elif kind == 'mixed':
        start = '2015-10-9'
        end = '2015-10-18'
    elif kind == 'combined':
        s = df.loc['2016-4-11':'2016-4-20']
        c = df.loc['2016-1-13':'2016-1-22']
        m = df.loc['2015-10-9':'2015-10-18']
        start = None
    if start is not None:
        output = df.loc[start:end]
    else:
        output = pd.concat([s, c, m])
        start = output.index[0]
        ts = pd.date_range(start.date(), periods=len(output), freq='5min')
        output.index = ts
    return output

def make_small_dev(df):
    cloudy = df.loc['2017-02-25':'2017-2-28']
    sunny = df.loc['2017-03-6':'2017-3-9']
    output = pd.concat([sunny, cloudy], axis=0)
    start = output.index[0]
    ts = pd.date_range(start.date(), periods=len(output), freq='5min')
    output.index = ts
    return output


if __name__ == "__main__":
    path_to_files = '/Users/bennetmeyers/Documents/CS229/Project/data_dump/'
    site_ids = '/Users/bennetmeyers/Documents/CS229/Project/SolarForecasting/data/selected_sites.txt'
    # summary = summarize_files(path_to_files, suffix='pkl', testing=True, verbose=True)
    # print summary
    df, keys = generate_master_dataset(site_ids, path_to_files, verbose=True)
    print(keys)
