# -*- coding: utf-8 -*-
"""
    scisys.validate
    ~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Tuple

import numpy as np
import pandas as pd
import logging

from copy import deepcopy
from corsys.tools import is_type

logger = logging.getLogger(__name__)


def validate(data: pd.DataFrame | pd.Series, method: str = None, inplace: bool = False) -> pd.DataFrame:
    validations = []

    for column, series in (data if isinstance(data, pd.DataFrame) else data.to_frame()).items():
        series_method = method
        if series_method is None:
            if is_type(series, 'time', 'progress', 'state', 'status', 'mode', 'code'):
                logger.debug(f'Skipping column "{column}": Unable to infer validation method')
                continue

            elif is_type(series, 'energy'):
                series_method = 'inc'
            else:
                series_method = 'std'
        if series_method == 'std':
            if series.index[-1] - series.index[0] < pd.Timedelta(days=7):
                continue
            validation, _ = find_error_std(series)
        elif series_method == 'inc':
            validation, _ = find_error_inc(series)
        else:
            raise ValueError(f"Unknown validation method: {method}")

        if inplace:
            print(validation)

        validations.append(validation)

    if len(validations) == 0:
        return pd.DataFrame()
    return pd.concat(validations, axis='columns').dropna(how='all')


def find_nan(data: pd.DataFrame | pd.Series, resolution: int) -> pd.DataFrame:
    data = deepcopy(data).dropna()
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Create list to hold info about each NaN block
    nan_blocks = []

    # Tag all occurrences of NaN in the data with True
    # (but not before first or after last actual entry)
    # data_nan.loc[:, 'NaN'] = data_nan.isna().any(axis=1)
    #
    # # First row of consecutive region is a True preceded by a False in NaN data_nan
    # gaps['start'] = data_nan.index[data_nan['NaN'] & ~data_nan['NaN'].shift(1).fillna(False)]
    #
    # # Last row of consecutive region is a False preceded by a True
    # gaps['end'] = data_nan.index[data_nan['NaN'] & ~data_nan['NaN'].shift(-1).fillna(False)]

    # Find nan periods longer than the resolution
    index_delta = pd.Series(data.index, index=data.index)
    index_delta = (index_delta - index_delta.shift(1))/pd.Timedelta(seconds=1)
    for index in index_delta.loc[index_delta > resolution].index:
        nan_blocks.append({'start': data.index[data.index.get_loc(index) - 1], 'end': index})

    nan_blocks = pd.DataFrame(nan_blocks, columns=['start', 'end'])
    if not nan_blocks.empty:
        nan_blocks = nan_blocks.sort_values('end').reset_index().drop('index', axis='columns')

        if nan_blocks['start'].dt.tz is None:
            nan_blocks['start'] = nan_blocks['start'].dt.tz_localize(data.index.tzinfo)
        if nan_blocks['end'].dt.tz is None:
            nan_blocks['end'] = nan_blocks['end'].dt.tz_localize(data.index.tzinfo)

        # How long is each region
        nan_blocks['span'] = nan_blocks['end'] - nan_blocks['start']
        nan_blocks['seconds'] = nan_blocks['span'] / pd.Timedelta(seconds=1)
        nan_blocks = nan_blocks[nan_blocks['seconds'] > resolution]
        nan_blocks.drop(columns=['span', 'seconds'], inplace=True)

    # Add error identification for future reconstruction
    nan_blocks['type'] = 'nan'

    return nan_blocks


def find_error_std(data: pd.DataFrame | pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = deepcopy(data).dropna()
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Create DataFrame to hold info about each error block
    error_blocks = pd.DataFrame()

    error_std = np.abs(data - data.mean()) > 3 * data.std()

    if not error_std.empty:
        # First row of consecutive region is a True preceded by a False in tags
        error_blocks['start'] = data[error_std & ~error_std.shift(1).fillna(False)].index

        # Last row of consecutive region is a False preceded by a True
        error_blocks['end'] = data[error_std & ~error_std.shift(-1).fillna(False)].index

        # Add error identification for future reconstruction
        error_blocks['type'] = 'error_std'

    return error_blocks, error_std


# noinspection PyTypeChecker
def find_error_inc(data: pd.DataFrame | pd.Series, tolerance: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_size = len(data.index)
    data = deepcopy(data).dropna()
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # Create DataFrame to hold info about each error block
    error_blocks = pd.DataFrame()

    error_inc = data < data.shift(1)

    if not error_inc.empty:
        for time in error_inc.replace(False, np.NaN).dropna().index:
            # index of element i from error_inc
            i = data.index.get_loc(time)
            if 2 <= i < data_size:
                error_flag = None

                # If a rounding or transmission error results in a single value being too big,
                # fix that single data point, else flag all decreasing values
                if data.iloc[i] >= data.iloc[i - 2] and not error_inc.iloc[i - 2]:
                    error_inc.iloc[i] = False
                    error_inc.iloc[i - 1] = True

                elif all(data.iloc[i - 1] > data.iloc[i:min(data_size - 1, i + tolerance)]):
                    error_flag = data.index[i]

                else:
                    j = i + 1
                    while j < data_size and data.iloc[i - 1] > data.iloc[j]:
                        if error_flag is None and j - i > tolerance:
                            error_flag = data.index[i]

                        error_inc.iloc[j] = True
                        j = j + 1

                if error_flag is not None:
                    logger.warning('Unusual behaviour at index %s for  %s', error_flag.strftime('%d.%m.%Y %H:%M'),
                                   data.any())

        # First row of consecutive region is a True preceded by a False in tags
        error_blocks['start'] = data.index[error_inc & ~error_inc.shift(1).fillna(False)]

        # Last row of consecutive region is a False preceded by a True
        error_blocks['end'] = data.index[error_inc & ~error_inc.shift(-1).fillna(False)]

        # Add error identification for future reconstruction
        error_blocks['type'] = 'error_inc'

    return error_blocks, error_inc
