# -*- coding: utf-8 -*-
"""
This module contains classes and functions for pre-processing data and initial data selection.
"""

from core.utilities import envelope_fit, masked_smooth_fit_periodic
import numpy as np
from numpy.linalg import svd
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from copy import copy

TRAIN = ('2015-7-15', '2016-11-25') # 500 days
DEV = ('2016-11-26', '2017-3-15')   # 110 days
TEST = ('2017-3-16', '2017-07-14')  # 121 days
DROP_LIST = [1, 2, 36, 37, 39, 41, 43, 65]

DEBUG = False       # Leave off unless debugging in an IDE

ORIGINAL_DATA = 'data/master_dataset.pkl'
DETRENDED_DATA = 'data/detrended_master_dataset.pkl'
CLEARSKY_DATA = 'data/clearsky_master_dataset.pkl'
if DEBUG:
    ORIGINAL_DATA = '../' + ORIGINAL_DATA
    DETRENDED_DATA = '../' + DETRENDED_DATA
    CLEARSKY_DATA = '../' + CLEARSKY_DATA

try:
    CLEARSKY_DF = pd.read_pickle(CLEARSKY_DATA)
except IOError:
    pass

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

    def make_clearsky_model(self, n=5, mu1=3.5, eta=1.5, mu2=3, plot=False, return_fits=False):
        if self.U is None:
            self.get_eigenvectors()
        daily_scale_factors = ((np.diag(self.D).dot(self.P[:288])))
        signals = []
        fits = np.empty((n, self.P.shape[1]))
        for ind in xrange(n):
            signal = daily_scale_factors[ind, :]
            if ind == 0:
                fit = envelope_fit(signal, mu=10**mu1, eta=10**eta, kind='lower', period=365)
                mask = np.abs(signal - fit) < 1.5
            else:
                mu_i = mu2
                fit = masked_smooth_fit_periodic(signal, mask, 365, mu=10**mu_i)
            signals.append(signal)
            fits[ind, :] = fit
        self.DP_clearsky = fits[:, :365]
        if plot:
            fig, axes = plt.subplots(nrows=n, figsize=(12,n*4), sharex=True)
            try:
                for ind in xrange(n):
                    axes[ind].plot(signals[ind], linewidth=1)
                    axes[ind].plot(fits[ind], linewidth=1)
                    axes[ind].set_title('Daily scale factors for eigenvector {}'.format(ind+1))
                axes[ind].set_xlabel('Day Number')
                plt.tight_layout()
                return fig, axes
            except TypeError:
                axes.plot(signals[0], linewidth=1)
                axes.plot(fits[0], linewidth=1)
                axes.set_xlabel('Day Number')
                axes.set_title('Daily scale factors for eigenvector 1')
                return fig, axes
        if return_fits:
            return signals, fits

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


def detrend_data(df, drop_bad_columns=True, return_clearsky=False):
    '''
    Detrend a datafrane of time-series power data using the StatisticalClearSky class
    :param df: A dataframe organized like the master dataset
    :param drop_bad_columns: Set True to drop columns observed to have corrupted data
    :return: A dataframe with daily and yearly periodic trends removed
    '''
    new_df = pd.DataFrame(index=df.index, columns=df.columns)
    for i in xrange(1, 75):
        if drop_bad_columns:
            flag = (i not in DROP_LIST)
        else:
            flag = True
        if flag:
            if i == 74:
                key = 'total_power'
            else:
                key = 'S{:02}'.format(i)
            scs = StatisticalClearSky(df[key])
            scs.make_clearsky_model(n=5, plot=False)
            clearsky = scs.estimate_clearsky(np.s_[:])
            clearsky = clearsky.ravel(order='F')
            extended = np.empty(len(df))
            clearsky = np.tile(clearsky, 1 + len(extended) / len(clearsky))
            extended = clearsky[:len(extended)]
            new_df[key] = extended
    detrended_data = new_df - df
    if return_clearsky:
        return detrended_data, new_df
    else:
        return detrended_data

def retrend_data(series, key='total_power', clearsky=None):
    '''
    This is a function for "retrending" forecasted data from a detreded data set. Requires clearsky data that was
     used to detrend the data. Think of this as the inverse transform.
    :param series: a Pandas series containing the data to retrend. Should have a time index that matches the clearsky
    data
    :param key: The column from the clearsky data to use for retrending
    :param clearsky: A dataframe containing the clearsky data. If left as None and CLEARSKY_DF can be loaded, the master
    detrended dataset will be used
    :return: a retrended time series
    '''
    if clearsky is None:
        try:
            clearsky = CLEARSKY_DF
        except NameError:
            print('Please provide a clearsky reference')
            return
    return clearsky[key].loc[series.index] - series


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

class DataManager(object):
    """
    A class for managing problem data. Use method load_all_and_split to execute all other methods at once. Leaving
    reindex set to False maintains original time stamps on the data, which for the small train and dev and not all
    sequential. This is the correct setting for analysis and exogenous data. Setting reindex to True is good for
    plotting.
    """
    def __init__(self):
        self.original_full = None
        self.original_train = None
        self.original_dev = None
        self.detrended_full = None
        self.detrended_train = None
        self.detrended_dev = None
        self.split_type = None
        self.reindexed = None
        self.forecasts = None

    def load_all_and_split(self, fp1=ORIGINAL_DATA, fp2=DETRENDED_DATA, kind='small', reindex=False):
        self.load_original_data(fp=fp1)
        self.load_detrended_data(fp=fp2)
        self.train_dev_split(kind=kind, reindex=reindex)

    def load_original_data(self, fp=ORIGINAL_DATA):
        df = pd.read_pickle(fp).fillna(0)
        df = df.loc['2015-07-15':'2017-07-14']
        self.original_full = df

    def load_detrended_data(self, fp=DETRENDED_DATA):
        df = pd.read_pickle(fp).fillna(0)
        self.detrended_full = df

    def train_dev_split(self, kind='small', reindex=False):
        self.split_type = kind
        self.reindexed = reindex
        if kind == 'small':
            if self.original_full is not None:
                train_df = make_small_train(self.original_full, kind='combined', reindex=reindex)
                dev_dv = make_small_dev(self.original_full, reindex=reindex)
                self.original_train = train_df
                self.original_dev = dev_dv
            if self.detrended_full is not None:
                train_df = make_small_train(self.detrended_full, kind='combined', reindex=reindex)
                dev_dv = make_small_dev(self.detrended_full, reindex=reindex)
                self.detrended_train = train_df
                self.detrended_dev = dev_dv
        elif kind == 'all':
            if self.original_full is not None:
                train_df, dev_df = train_dev_test_split(self.original_full, train=TRAIN, dev=DEV, test=None)
                self.original_train = train_df
                self.original_dev = dev_dv
            if self.detrended_full is not None:
                train_df, dev_df = train_dev_test_split(self.detrended_full, train=TRAIN, dev=DEV, test=None)
                self.detrended_train = train_df
                self.detrended_dev = dev_dv

    def add_forecasts(self, forecasts):
        self.forecasts = forecasts

    def swap_index(self, include_forecasts=True):
        if self.reindexed == False and self.split_type == 'small':
            self.reindexed = True
            if self.original_train is not None:
                start = self.original_train.index[0]
                ts_train = pd.date_range(start.date(), periods=len(self.original_train), freq='5min')
                self.original_train.index = ts_train
                start = self.original_dev.index[0]
                ts_dev = pd.date_range(start.date(), periods=len(self.original_dev), freq='5min')
                ts_dev_old = self.original_dev.index
                self.original_dev.index = ts_dev
            if self.detrended_train is not None:
                start = self.detrended_train.index[0]
                ts_train = pd.date_range(start.date(), periods=len(self.detrended_train), freq='5min')
                self.detrended_train.index = ts_train
                start = self.detrended_dev.index[0]
                ts_dev = pd.date_range(start.date(), periods=len(self.detrended_dev), freq='5min')
                ts_dev_old = self.detrended_dev.index
                self.detrended_dev.index = ts_dev
            if include_forecasts:
                columns = ['f{:02}'.format(n) for n in xrange(len(self.forecasts))]
                cols = np.r_[['new_index'], columns]
                temp = pd.DataFrame(index=ts_dev_old, columns=cols)
                temp['new_index'] = ts_dev
                for n, f in enumerate(self.forecasts):
                    key = 'f{:02}'.format(n)
                    try:
                        temp.loc[f.index, [key]] = f
                    except KeyError:
                        # Only occurs for nighttime forecasts at the very end of dev period, which are all zero and
                        # we don't actually care about
                        del temp[key]
                        columns.remove(key)
                temp.set_index('new_index', inplace=True)
                new_forecasts = [temp[col].dropna() for col in columns]
                self.forecasts = new_forecasts
        elif self.reindexed == True and self.split_type == 'small':
            self.reindexed = False
            s = pd.date_range('2016-4-11', '2016-4-21', freq='5min')[:-1]
            c = pd.date_range('2016-1-13', '2016-1-23', freq='5min')[:-1]
            m = pd.date_range('2015-10-9', '2015-10-19', freq='5min')[:-1]
            ts_train = s.append(c.append(m))
            c = pd.date_range('2017-02-25', '2017-3-1', freq='5min')[:-1]
            s = pd.date_range('2017-03-6', '2017-3-10', freq='5min')[:-1]
            ts_dev = s.append(c)
            if self.original_train is not None:
                ts_dev_old = self.original_dev.index
                self.original_train.index = ts_train
                self.original_dev.index = ts_dev
            if self.detrended_train is not None:
                ts_dev_old = self.detrended_dev.index
                self.detrended_train.index = ts_train
                self.detrended_dev.index = ts_dev
            if include_forecasts:
                columns = ['f{:02}'.format(n) for n in xrange(len(self.forecasts))]
                cols = np.r_[['new_index'], columns]
                temp = pd.DataFrame(index=ts_dev_old, columns=cols)
                temp['new_index'] = ts_dev
                for n, f in enumerate(self.forecasts):
                    key = 'f{:02}'.format(n)
                    try:
                        temp.loc[f.index, [key]] = f
                    except KeyError:
                        # Only occurs for nighttime forecasts at the very end of dev period, which are all zero and
                        # we don't actually care about
                        del temp[key]
                        columns.remove(key)
                temp.set_index('new_index', inplace=True)
                new_forecasts = [temp[col].dropna() for col in columns]
                self.forecasts = new_forecasts


def make_index_sequential(df):
    start = df.index[0]
    ts = pd.date_range(start.date(), periods=len(df), freq='5min')
    df.index = ts
    return df

def make_small_train(df, kind='mixed', reindex=True):
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
        if reindex is True:
            output = make_index_sequential(output)
    return output

def make_small_dev(df, reindex=True):
    cloudy = df.loc['2017-02-25':'2017-2-28']
    sunny = df.loc['2017-03-6':'2017-3-9']
    output = pd.concat([sunny, cloudy], axis=0)
    if reindex:
        output = make_index_sequential(output)
    return output

def make_batch(df, size, present, future):
    """
    Takes a dataframe and produces batches of given size.

    Parameters:
        size    - batch size
        present - number of time steps from each inverter
        future  - number of time steps to predict in the aggregate
    """
    features = df.iloc[:,0:-1] # assume inverters are in columns 2, 3, ..., n-1
    response = df.iloc[:,-1] # assume aggregate power is in column n

    n = df.shape[0]

    X = []; Y = []
    D = []
    for i in range(size):
        t = np.random.randint(n - present - future)
        x = features[t:t+present].values.T.flatten().tolist()
        y = response[t+present:t+present+future].values.tolist()
        day = features.index.dayofyear[t]
        X.append(x)
        Y.append(y)
        D.append(day)

    # convert to Numpy
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    D = np.array(D, dtype=np.float32)
    D = D[:,np.newaxis]
    return X, Y, D

def center_design_matrix(X):
    """
    Center the rows of the design matrix.
    """
    m = np.mean(X, 0)
    s = np.std(X, 0)
    Xcent = (X - m) / s
    return Xcent


if __name__ == "__main__":
    #path_to_files = '/Users/bennetmeyers/Documents/CS229/Project/data_dump/'
    #site_ids = '/Users/bennetmeyers/Documents/CS229/Project/SolarForecasting/data/selected_sites.txt'
    #path_to_data = '../Data/master_dataset.pkl'
    #df = pd.read_pickle(path_to_data).fillna(0)
    #df = df.loc['2015-07-15':'2017-07-14']
    # summary = summarize_files(path_to_files, suffix='pkl', testing=True, verbose=True)
    # print summary
    # df, keys = generate_master_dataset(site_ids, path_to_files, verbose=True)
    #detrend_data(df)
    from core.arima_models import SumToSumARIMA
    dm = DataManager()
    dm.load_all_and_split(reindex=True)
    prob = SumToSumARIMA(df=dm.detrended_train)
    prob.train(order=(1, 1, 0))
    prob.test(dm.detrended_dev['total_power'])
    dm.add_forecasts(prob.forecasts)
    dm.swap_index()
    transformed_forecasts = [retrend_data(f) for f in dm.forecasts]
    dm.add_forecasts(transformed_forecasts)
    dm.swap_index()
    from utilities import calc_test_mse
    calc_test_mse(dm.original_dev, dm.forecasts)
