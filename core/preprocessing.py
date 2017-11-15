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
        df_view = df[df['serial_number'] == sn].dropna()
        tstart = df.iloc[0]
        tend = df.iloc[-1]
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
            'stdev': ac_stdev
        }
    return output

def load_raw_file(filename):
    '''
    Might work on other types on SunPower files. Probably won't do the right thing on ther files. Handles CSVs and
    compressed CSVs, including .gz and .zip
    :param filename: the file path to a SunPower type-130 raw data file
    :return: a pandas
    '''
    df = pd.read_csv(files[0], index_col=False, parse_dates=[1])
    return df

def summarize_files(file_path, suffix='gz', verbose=False):
    '''

    :param file_path: a string containing the path to the directory containing the files to be summarized
    :param type: options are 'gz', 'zip', or 'csv'
    :return: a pandas dataframe with the summary of the files
    '''
    if not file_path[-1] == '/':
        file_path += '/'
    search = file_path + '*.' + suffix
    files = glob(search)
    data = {}
    N = len(files)
    if verbose:
        print '{} files to process'.format(N)
    for it, fn in enumerate(files):
        df = load_raw_file(fn)
        summary = data_summary(df)
        data[fn] = summary
        if verbose:
            print '{}/{} complete:'.format(it+1, N), fn.split('/')[-1]
    df = pd.DataFrame(data=data)
    return df.T
