# -*- coding: utf-8 -*-
"""
    scisys.process
    ~~~~~~~~~~~~~~
    
    
"""
from __future__ import annotations

import os
import pytz as tz
import numpy as np
import pandas as pd
import datetime as dt
import warnings
import logging

from typing import Union
from copy import deepcopy

# noinspection PyProtectedMember
from corsys.tools import _resample_series, to_date, floor_date, ceil_date
from corsys.system import System
from corsys.cmpt import Photovoltaics

logger = logging.getLogger(__name__)


# noinspection PyShadowingBuiltins
def process_lpg(key=None, dir='LPG',
                hot_water_factor=2793.3,
                timezone: tz.timezone = tz.timezone('Europe/Berlin'), sep=';', **_) -> pd.DataFrame:
    if key is not None:
        dir = os.path.join(dir, key)

    if not os.path.isdir(dir):
        raise Exception("Unable to access LPG directory: {0}".format(dir))

    def read(file: str, index='Time'):
        return pd.read_csv(os.path.join(dir, 'Results', file), skipinitialspace=True, low_memory=False, sep=sep,
                           index_col=[index], parse_dates=[index])

    data_el = read('SumProfiles.Electricity.csv')
    data_el = data_el[['Sum [kWh]']].rename(columns={'Sum [kWh]': 'el_energy_delta'})

    # data_th_ht = read('SumProfiles.Space Heating.csv')
    # data_th_ht = data_th_ht[['Sum [kWh]']].rename(columns={'Sum [kWh]': 'th_ht_energy_delta'})
    data_th_ht = read('DeviceProfiles.House.Space Heating.csv')
    data_th_ht = data_th_ht[['House - Space Heating Location - Space Heating [kWh]']]\
        .rename(columns={'House - Space Heating Location - Space Heating [kWh]': 'th_ht_energy_delta'})

    data_th_dom = read('SumProfiles.Hot water.csv')
    data_th_dom = data_th_dom[['Sum [L]']].rename(columns={'Sum [L]': 'th_ht_liters'})

    data = pd.concat([data_el, data_th_ht, data_th_dom], axis=1)

    data.index.name = 'time'
    data.index = data.index.tz_localize(timezone, ambiguous="NaT", nonexistent="NaT")
    data = data[pd.notnull(data.index)]

    data_res = data.index[1] - data.index[0]
    data_time = pd.DataFrame(index=data.index, data=data.index)
    data_time.columns = ['date']
    data_time['hours'] = ((data_time['date'] - data_time['date'].shift(1)) / np.timedelta64(1, 'h')).bfill()

    data[System.POWER_EL] = data['el_energy_delta']/data_time['hours']*1000
    data[System.POWER_TH_HT] = data['th_ht_energy_delta']/data_time['hours']*1000

    # Calculate domestic water consumption from liters and process with a rolling average of 10 minutes
    data[System.POWER_TH_DOM] = data['th_ht_liters']*hot_water_factor
    data[System.POWER_TH_DOM] = data[System.POWER_TH_DOM].rolling(window=int(600/data_res.seconds)).mean()
    data[System.POWER_TH_DOM] = data[System.POWER_TH_DOM].rolling(window=int(600/data_res.seconds),
                                                                  win_type="gaussian", center=True).mean(std=20)\
                                                                                                   .fillna(0)

    data[System.POWER_TH] = data[System.POWER_TH_HT] + data[System.POWER_TH_DOM]

    data[System.ENERGY_EL] = data['el_energy_delta'].cumsum()
    data[System.ENERGY_TH_HT] = data['th_ht_energy_delta'].cumsum()
    data[System.ENERGY_TH_DOM] = (data[System.POWER_TH_DOM]/1000*data_time['hours']).cumsum()
    data[System.ENERGY_TH] = (data[System.POWER_TH]/1000*data_time['hours']).cumsum()

    return data[[System.POWER_EL, System.POWER_TH, System.POWER_TH_HT, System.POWER_TH_DOM,
                 System.ENERGY_EL, System.ENERGY_TH, System.ENERGY_TH_HT, System.ENERGY_TH_DOM]]


# noinspection PyShadowingBuiltins
def process_opsd(key: str = None, dir: str = 'OPSD', **_) -> pd.DataFrame:
    if not os.path.isdir(dir):
        raise Exception("Unable to access OPSD directory: {0}".format(dir))

    index = 'utc_timestamp'
    data = pd.read_csv(os.path.join(dir, 'household_data_1min.csv'),
                       skipinitialspace=True, low_memory=False, sep=',',
                       index_col=[index], parse_dates=[index])

    if key is None:
        raise Exception("Unable to process OPSD with unconfigured key")

    data.index.rename('time', inplace=True)
    data = data.filter(regex=key)  # .dropna(how='all')
    if data.empty:
        raise Exception("Unable to find OPSD household: " + key)

    for column in data.columns:
        column_name = column.split(key + '_', 1)[1] + '_energy'
        data.rename(columns={column: column_name}, inplace=True)

    columns_power = [System.POWER_EL, System.POWER_EL_IMP]
    columns_energy = [System.ENERGY_EL, System.ENERGY_EL_IMP]
    data[System.ENERGY_EL_IMP] = _process_energy(data['grid_import_energy'])
    data[System.POWER_EL_IMP] = _process_power(data['grid_import_energy'])

    if 'grid_export_energy' in data.columns:
        columns_power.append(System.POWER_EL_EXP)
        columns_energy.append(System.ENERGY_EL_EXP)
        data[System.ENERGY_EL_EXP] = _process_energy(data['grid_export_energy'])
        data[System.POWER_EL_EXP] = _process_power(data['grid_export_energy'])

    data_pv = data[[c for c in data.columns if c.startswith('pv_')]]
    if not data_pv.empty:
        for column_energy in data_pv.columns:
            column_power = column_energy.replace('_energy', '_power')
            columns_power.append(column_power)
            columns_energy.append(column_energy)
            data[column_energy] = _process_energy(data_pv[column_energy])
            data[column_power] = _process_power(data_pv[column_energy])

        if 'pv_energy' not in data.columns:
            columns_power.append(Photovoltaics.POWER)
            columns_energy.append(Photovoltaics.ENERGY)
            data[Photovoltaics.ENERGY] = _process_energy(data_pv.fillna(0).sum(axis=1))
            data[Photovoltaics.POWER] = _process_power(data[Photovoltaics.ENERGY])

    data[System.ENERGY_EL] = data[System.ENERGY_EL_IMP]
    if 'pv_energy' in data.columns:
        pv_cons = data[Photovoltaics.ENERGY]
        if 'grid_export_energy' in data.columns:
            pv_cons -= data[System.ENERGY_EL_EXP]
        data[System.ENERGY_EL] += pv_cons

    if 'heat_pump_energy' in data.columns:
        columns_power += [System.POWER_TH, 'hp_power']
        columns_energy += [System.ENERGY_TH, 'hp_energy']

        data['hp_energy'] = _process_energy(data['heat_pump_energy'])
        data['hp_power'] = _process_power(data['heat_pump_energy'])

        data[System.ENERGY_EL] -= data['hp_energy']

        # TODO: Make COP more sophisticated
        # Maybe try to differentiate between heating and warm water
        cop = 3.5
        data[System.POWER_TH] = _process_power(data['heat_pump_energy']) * cop  # , filter=False)

        # Offset and widening of thermal power from heat pump power, smoothen peaks and reduce offset again
        data_back = data[System.POWER_TH].iloc[::-1]
        data_back = data_back.rolling(window=200).mean()
        data_front = data_back.rolling(window=50, win_type="gaussian", center=True).mean(std=15).iloc[::-1]
        data[System.POWER_TH] = data_front.rolling(window=150).mean().ffill().bfill()

        data_time = pd.DataFrame(index=data.index, data=data.index)
        data_time.columns = ['date']
        data_time['hours'] = ((data_time['date'] - data_time['date'].shift(1)) / np.timedelta64(1, 'h')).bfill()

        data[System.ENERGY_TH] = (data[System.POWER_TH] / 1000 * data_time['hours']).fillna(0).cumsum()

    data[System.POWER_EL] = _process_power(data[System.ENERGY_EL])

    return data[columns_power]  # + columns_energy]


# noinspection PyProtectedMember
# noinspection PyShadowingBuiltins
def process_meteoblue(dir: str = 'Meteoblue',
                      latitude: str | float = None,
                      longitude: str | float = None,
                      start: str | pd.Timestamp | dt.datetime = None,
                      end:   str | pd.Timestamp | dt.datetime = None, **_) -> pd.DataFrame:
    from corsys.io._var import WEATHER

    if latitude is None or longitude is None:
        raise Exception("Unable to process meteoblue data for unconfigured latitude or longitude")

    latitude = latitude if isinstance(latitude, float) else float(latitude)
    longitude = longitude if isinstance(longitude, float) else float(longitude)

    location = '{0:06.2f}'.format(latitude).replace('.', '') + '_' + '{0:06.2f}'.format(longitude).replace('.', '')
    location_dir = os.path.join(dir, 'Locations', location)
    if not os.path.isdir(dir):
        raise Exception("Unable to access meteoblue directory: {0}".format(dir))

    data_info = []
    data_content = []

    for entry in os.scandir(dir):
        if entry.is_file() and entry.path.endswith('.csv'):
            info = pd.read_csv(entry.path, skipinitialspace=True, low_memory=False, sep=';',
                               header=None, index_col=[0]).iloc[:18, :]
            info.columns = info.iloc[3]

            data_info.append(info.loc[:, ~info.columns.duplicated()].dropna(axis=1, how='all'))
            data_content.append(pd.read_csv(entry.path, skipinitialspace=True, low_memory=False, sep=';',
                                            header=[18], index_col=[0, 1, 2, 3, 4]))

    points = pd.concat(data_info, axis=0).drop_duplicates()
    histories = pd.concat(data_content, axis=1)
    for point in points.columns.values:
        if abs(latitude - float(points.loc['LAT', point]) > 0.001) or \
           abs(longitude - float(points.loc['LON', point]) > 0.001):
            continue

        columns = [column for column in histories.columns.values if column.startswith(point + ' ')]
        data = histories[columns].copy()
        data.columns = [c.replace(c.split(' ')[0], '').replace(c.split('[')[1], '').replace('  [', '') for c in columns]
        data['time'] = [dt.datetime(y, m, d, h, n) for y, m, d, h, n in data.index]
        data.set_index('time', inplace=True)
        data.index = data.index.tz_localize(tz.utc)
        # data.index = history.index.tz_convert('Europe/Berlin')
        data = data.rename(columns={' Temperature':             'temp_air',
                                    ' Wind Speed':              'wind_speed',
                                    ' Wind Gust':               'wind_speed_gust',
                                    ' Wind Direction':          'wind_direction',
                                    ' Relative Humidity':       'relative_humidity',
                                    ' Mean Sea Level Pressure': 'pressure_sea',
                                    ' Shortwave Radiation':     'ghi',
                                    ' DNI - backwards':         'dni',
                                    ' DIF - backwards':         'dhi',
                                    ' Total Cloud Cover':       'cloud_cover',
                                    ' Low Cloud Cover':         'clouds_low',
                                    ' Medium Cloud Cover':      'clouds_mid',
                                    ' High Cloud Cover':        'clouds_high',
                                    ' Total Precipitation':     'precipitation',
                                    ' Snow Fraction':           'snow_fraction'})

        if start is not None or end is not None:
            if start is not None:
                start = to_date(start)
            if end is not None:
                end = to_date(end)
                end = ceil_date(end)

        if os.path.isdir(location_dir) and (end is None or end > data.index[-1]):
            # Delete unavailable column of continuous forecasts
            del data['wind_speed_gust']

            for file in sorted(os.listdir(location_dir)):
                path = os.path.join(location_dir, file)
                if os.path.isfile(path) and file.endswith('.csv'):
                    forecast = pd.read_csv(path, index_col='time', parse_dates=['time'])
                    forecast = forecast.rename(columns={'humidity_rel': 'relative_humidity',
                                                        'rain':         'precipitation',
                                                        'rain_shower':  'precipitation_convective',
                                                        'rain_prob':    'precipitation_probability',
                                                        'snow':         'snow_fraction'})

                    start = forecast.index[0]
                    data = forecast.loc[start:start+dt.timedelta(hours=23, minutes=59, seconds=59), data.columns]\
                                   .combine_first(data)

                    # if os.path.exists(weather_lib):
                    #     loc_file = os.path.join(weather_lib, time.strftime('%Y%m%d_%H%M%S') + '.csv')
                    #     if not os.path.exists(loc_file):
                    #         forecast.to_csv(loc_file, sep=',', encoding='utf-8-sig')

        # data_index = pd.date_range(start=data.index[0],
        #                            end=data.index[-1],
        #                            freq=str((data.index[1] - data.index[0]).seconds)+'s')
        # data = data.combine_first(pd.DataFrame(index=data_index, columns=data.columns))
        # data.index.name = 'time'

        data = data[[column for column in WEATHER.keys() if column in data.columns]]

        # Upsample forecast to a resolution of 1 minute. Use the advanced Akima interpolator for best results
        data = data.resample('1Min').interpolate(method='akima')

        for irr in ['ghi', 'gni', 'dni', 'dhi', 'etr']:
            if irr in data.columns:
                data[data[irr] < 0] = 0
            if irr+'_instant' in data.columns:
                data[data[irr+'_instant'] < 0] = 0

        if 'snow_fraction' in data.columns:
            data.snow_fraction = data.snow_fraction.round()

    return data


def process(data: pd.DataFrame, resolution: int = 1, **kwargs) -> pd.DataFrame:
    series = []
    columns = data.columns
    for column in columns:
        series_processed = _process_series(data[column], resolution, **kwargs)
        series.append(series_processed)
    data = pd.concat(series, axis=1).dropna(how='all')
    return data[columns]


def _process_series(data: pd.Series, resolution: int = 1, fill_gaps: bool = False, **kwargs) -> pd.Series:
    # Find measurement outages longer than the resolution
    # TODO: configure (dynamic) maximum logging interval and use for gap location
    gaps = _locate_gaps(data, 1)

    # Extend index to have a regular frequency
    start_minute = data.index[0].minute
    if start_minute > 0:
        start_minute += resolution - start_minute % resolution
    start_hour = data.index[0].hour
    if start_minute > 59:
        start_minute = 0
        start_hour += 1
    start = data.index[0].replace(hour=start_hour, minute=start_minute, second=0)

    end_minute = data.index[-1].minute - (data.index[-1].minute % resolution)
    end = data.index[-1].replace(minute=end_minute, second=59)

    logger.debug('Processing data series "%s" from %s to %s', data.name, start, end)
    timezone = data.index.tzinfo
    index = pd.date_range(start=start, end=end, tz=timezone, freq='{}min'.format(resolution))
    data_name = data.name
    data = data.to_frame().combine_first(pd.DataFrame(index=index, columns=[data_name]))
    data.index.name = 'time'

    # Drop rows with outages longer than the resolution
    for _, gap in gaps.iterrows():
        error = data[(data.index > gap['start']) & (data.index < gap['end'])]
        data = data.drop(error.index)

    # Interpolate the values between the irregular data points and drop them afterwards,
    # to receive a regular index that is sure to be continuous, in order to later expose
    # remaining gaps in the data. Use the advanced Akima interpolator for best results
    data = data.interpolate(method='akima').bfill()
    data = _resample_series(data[data_name], resolution*60)

    if fill_gaps:
        data = _impute(data, gaps, **kwargs)

    return data[start:end]


def _process_energy(energy: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    energy = energy.fillna(0)
    return energy - energy[0]


# noinspection PyShadowingBuiltins
def _process_power(energy: pd.DataFrame | pd.Series, filter: bool = True) -> pd.DataFrame | pd.Series:
    delta_energy = energy.diff()
    delta_index = pd.Series(energy.index, index=energy.index)
    delta_index = (delta_index - delta_index.shift(1))/np.timedelta64(1, 'h')

    data_power = (delta_energy/delta_index).fillna(0)*1000

    if filter:
        from scipy import signal
        b, a = signal.butter(1, 0.25)
        data_power = signal.filtfilt(b, a, data_power, method='pad', padtype='even', padlen=15)
        data_power[data_power < 0.1] = 0

    return data_power


def _process_gaps(data: pd.DataFrame | pd.Series, **kwargs) -> pd.DataFrame | pd.Series:
    res = (data.index[1] - data.index[0]).total_seconds()/60
    gaps = _locate_gaps(data, res)
    data = _impute(data, gaps, **kwargs)

    return data


def _locate_gaps(data: pd.DataFrame | pd.Series, resolution) -> pd.DataFrame:
    # Create DataFrame to hold info about each NaN block
    gaps = pd.DataFrame()

    data_nan = deepcopy(data).dropna()
    if isinstance(data_nan, pd.Series):
        data_nan = data_nan.to_frame()

    # Tag all occurrences of NaN in the data with True
    # (but not before first or after last actual entry)
    # data_nan.loc[:, 'NaN'] = data_nan.isna().any(axis=1)
    #
    # # First row of consecutive region is a True preceded by a False in NaN data_nan
    # gaps['start'] = data_nan.index[data_nan['NaN'] & ~data_nan['NaN'].shift(1).fillna(False)]
    #
    # # Last row of consecutive region is a False preceded by a True
    # gaps['end'] = data_nan.index[data_nan['NaN'] & ~data_nan['NaN'].shift(-1).fillna(False)]

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)

        # Find measurement outages longer than the resolution
        index_delta = pd.Series(data_nan.index, index=data_nan.index)
        index_delta = (index_delta - index_delta.shift(1))/np.timedelta64(1, 'm')
        for index in index_delta.loc[index_delta > resolution].index:
            gaps = gaps.append({'start': data_nan.index[data_nan.index.get_loc(index) - 1],
                                'end': index}, ignore_index=True)

    gaps = gaps.sort_values('end').reset_index()
    gaps.drop('index', axis=1, inplace=True)
    if gaps['start'].dt.tz is None:
        gaps['start'] = gaps['start'].dt.tz_localize(data_nan.index.tzinfo)
    if gaps['end'].dt.tz is None:
        gaps['end'] = gaps['end'].dt.tz_localize(data_nan.index.tzinfo)

    # How long is each region
    gaps['span'] = gaps['end'] - gaps['start'] + dt.timedelta(minutes=resolution)
    gaps['minutes'] = gaps['span'] / dt.timedelta(minutes=resolution)

    return gaps[gaps['minutes'] > resolution]


def _impute(data: pd.DataFrame | pd.Series,
            data_nan: pd.DataFrame, days_prior: int = 365, **_) -> pd.DataFrame | pd.Series:
    for i, span_nan in data_nan.iterrows():
        # Interpolate missing value spans up to 1 hour
        if span_nan['span'] <= dt.timedelta(hours=1):
            continue

        data = _impute_by_day(data, span_nan, i, days_prior)

    return data


def _impute_by_day(data: pd.DataFrame | pd.Series,
                   data_nan: pd.Series, i: int, days_prior: int) -> pd.DataFrame | pd.Series:
    # Get the frequency/length of one period of data
    data_res = data.index[1] - data.index[0]

    start = data_nan['start']
    end = data_nan['end']
    if (end - start).days > days_prior:
        end = start + dt.timedelta(days=days_prior) - data_res

    while end <= data_nan['end']:
        days_offset = days_prior
        while data[start:end].isnull().values.any():
            if start - dt.timedelta(days=days_offset) < data.index[0]:
                if days_prior > 1:
                    logger.debug("Problem filling %i. gap for %i prior days. Attempting with %i prior days.",
                                 i + 1, days_prior, days_prior - 1)

                    days_prior -= 1
                    days_offset = days_prior
                    if (end - start).days > days_prior:
                        end = start + dt.timedelta(days=days_prior) - data_res

                    continue

                break

            elif end - dt.timedelta(days=days_offset) > data_nan['start']:
                days_offset += days_prior
                continue

            data_fill = data[start - dt.timedelta(days=days_offset) - data_res:
                             end - dt.timedelta(days=days_offset) + data_res]

            if data_fill.isnull().values.any():
                logger.debug("Problem filling %i. gap with data from %s to %s", i + 1,
                             data_fill.index[0], data_fill.index[-1])
                days_offset += days_prior
                continue

            data_fill.index += dt.timedelta(days=days_offset)
            data[start:end] = data_fill

            days_offset += days_prior

        if end == data_nan['end']:
            break

        start += dt.timedelta(days=days_prior)
        end += dt.timedelta(days=days_prior)
        if end > data_nan['end']:
            end = data_nan['end']

    if data[data_nan['start']:data_nan['end']].isnull().values.any():
        logger.warning("Unable to fill %i. gap from %s to %s", i + 1,
                       data_nan['start'], data_nan['end'])

    return data
