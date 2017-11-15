# -*- coding: utf-8 -*-
"""
This module contains classes and functions for pre-processing data and initial data selection.
"""

import numpy as np
import pandas as pd
from glob import glob

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
    print '{} files to process'.format(N)
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
            print '{}/{} complete:'.format(it+1, N), name
    df = pd.DataFrame.from_dict(data, orient='index')
    df = df[['t_start', 't_end', 'num_vals', 'ac_max', 'ac_min', 'ac_avg', 'ac_stdev']]
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
        print '{} files to process'.format(N)
    for it, fn in enumerate(files):
        df = load_raw_file(fn)
        name = fn.split('/')[-1]
        name = name.split('.')[0]
        save_path = file_path + name + '.pkl'
        df.to_pickle(save_path)
        if verbose:
            print '{}/{} complete:'.format(it+1, N), name

if __name__ == "__main__":
    path_to_files = '/Users/bennetmeyers/Documents/CS229/Project/data_dump/'
    summary = summarize_files(path_to_files, suffix='pkl', testing=True, verbose=True)
    print summary