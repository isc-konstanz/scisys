# -*- coding: utf-8 -*-
"""
    scisys.build
    ~~~~~~~~~~~~
    
    
"""
from __future__ import annotations
from typing import Optional

import logging
import numpy as np
import pandas as pd
import datetime as dt
from corsys import System, Location
from corsys.io import Database
from corsys.configs import Configurations
from corsys.tools import ceil_date, to_date, to_int, to_bool

logger = logging.getLogger(__name__)


# noinspection PyShadowingBuiltins, SpellCheckingInspection
def build(configs: Configurations,
          database: Database,
          start: str | pd.Timestamp | dt.datetime = None,
          end:   str | pd.Timestamp | dt.datetime = None, **kwargs) -> Optional[pd.DataFrame]:

    if database is None or not database.enabled or not configs.has_section('Data'):
        return None

    buildargs = dict(configs.items('Data'))
    rename = to_bool(buildargs.pop('rename', False))
    split = to_bool(buildargs.pop('split', True))

    start = to_date(start, timezone=database.timezone)
    end = to_date(end, timezone=database.timezone)
    end = ceil_date(end)
    if start is None and end is None and 'year' in buildargs:
        from dateutil.relativedelta import relativedelta
        start = pd.Timestamp(to_int(buildargs.pop('year')), 1, 1).tz_localize(database.timezone)
        end = start + relativedelta(years=1) - dt.timedelta(seconds=1)

    buildargs['start'] = start
    buildargs['end'] = end
    buildargs.update(kwargs)

    if database.exists(**buildargs):
        return database.read(**buildargs)

    type = configs.get('Data', 'type', fallback=None).strip()
    # types = configs.get('Data', 'type', fallback='default')
    # for type in types.split(','):
    #     type = type.strip().lower()

    if start is not None and end is not None:
        logger.info('Building %s data from %s to %s', type,
                    start.strftime('%d.%m.%Y %H:%M'),
                    end.strftime('%d.%m.%Y %H:%M'))
    else:
        logger.info('Building %s data', type)

    if type.lower() in 'lpg':
        data = build_lpg(**buildargs)
    elif type.lower() == 'opsd':
        data = build_opsd(**buildargs)
    elif type.lower() == 'meteoblue':
        data = build_meteoblue(**buildargs)
    elif type.lower() == 'brightsky':
        data = build_brightsky(**buildargs)
    else:
        raise ValueError('Invalid data build type: {}'.format(type))

    if database.enabled and data is not None and not data.empty:
        if start is None:
            start = data.index[0]
        if end is None:
            end = data.index[-1]
        data = data[(data.index >= start) &
                    (data.index <= end)]
        if data.index.tzinfo is not None and data.index.tzinfo.utcoffset(data.index) is not None:
            data = data.tz_convert(database.timezone)

        database.write(data, split_data=split, rename=rename, **kwargs)

    return data


def build_lpg(key: str = None, weather: pd.DataFrame = None, **kwargs) -> pd.DataFrame:
    from scisys.process import process_lpg

    data = pd.DataFrame()
    if key is None:
        raise ValueError('The LPG key needs to be configured')

    for k in key.split(','):
        d = process_lpg(key=k.strip(), **kwargs)
        if d is not None and not d.empty:
            data = data.combine_first(d)

    # TODO: iterate full years and calculate annual heating demand
    data_time = pd.DataFrame(index=data.index, data=data.index)
    data_time.columns = ['date']
    data_time['hours'] = ((data_time['date'] - data_time['date'].shift(1)) / np.timedelta64(1, 'h')).bfill()

    data_th_ht = _build_oemof(data.loc[data.index[-1], System.ENERGY_TH_HT], weather, **kwargs)

    data[System.POWER_TH_HT] = (data_th_ht/data_time['hours']*1000).interpolate(method='akima')

    data[System.POWER_TH] = data[System.POWER_TH_HT] + data[System.POWER_TH_DOM]
    data[System.ENERGY_TH] = (data[System.POWER_TH]/1000*data_time['hours']).cumsum()
    data[System.ENERGY_TH_HT] = data_th_ht.cumsum()

    return data


# noinspection PyShadowingBuiltins, PyPackageRequirements, SpellCheckingInspection
def _build_oemof(annual_demand: float,
                 weather: pd.DataFrame,
                 country: str = 'DE/BW',
                 building_type: str = "EFH",
                 building_class: int = 1,
                 wind_class: int = 1, **_) -> pd.Series:
    import demandlib.bdew as bdew
    import holidays as hl

    # TODO: sanitize country parsing
    country = country.split('/')
    years = list(dict.fromkeys(weather.index.year))
    holidays = hl.country_holidays(country[0], subdiv=country[1], years=years)

    data = bdew.HeatBuilding(weather.index,
                             holidays=holidays,
                             temperature=weather['temp_air'],
                             annual_heat_demand=annual_demand,
                             shlp_type=building_type,
                             building_class=int(building_class),
                             wind_class=int(wind_class),
                             ww_incl=False).get_bdew_profile()

    data_res = int(3600/(data.index[1] - data.index[0]).seconds)
    data = data.rolling(window=data_res, win_type="gaussian", center=True).mean(std=20).fillna(data)

    return data


def build_opsd(**kwargs) -> pd.DataFrame:
    from scisys.process import process_opsd
    return process_opsd(**kwargs)


def build_meteoblue(location: Location, **kwargs) -> pd.DataFrame:
    if 'latitude' not in kwargs:
        kwargs['latitude'] = location.latitude
    if 'longitude' not in kwargs:
        kwargs['longitude'] = location.longitude

    from scisys.process import process_meteoblue
    return process_meteoblue(**kwargs)


def build_brightsky(location: Location,
                    start: str | pd.Timestamp | dt.datetime,
                    end: str | pd.Timestamp | dt.datetime,
                    address: str = 'https://api.brightsky.dev/', **_) -> pd.DataFrame:
    import json
    import requests
    from corsys.weather import Weather

    date = start
    date_last = end + dt.timedelta(days=1)

    parameters = {
        'date': date.strftime('%Y-%m-%d'),
        'last_date': date_last.strftime('%Y-%m-%d'),
        'lat': location.latitude,
        'lon': location.longitude,
        'tz': location.timezone.zone
    }
    response = requests.get(address + 'weather', params=parameters)

    if response.status_code != 200:
        raise requests.HTTPError("Response returned with error " + str(response.status_code) + ": " +
                                 response.reason)

    data = json.loads(response.text)
    data = pd.DataFrame(data['weather'])
    data['timestamp'] = pd.DatetimeIndex(pd.to_datetime(data['timestamp'], utc=True))
    data = data.set_index('timestamp').tz_convert(location.timezone)
    data.index.name = 'time'

    if data[Weather.CLOUD_COVER].isna().any():
        data[Weather.CLOUD_COVER].interpolate(method='linear', inplace=True)

    data.rename(columns={
        'solar': Weather.GHI,
        'temperature': Weather.TEMP_AIR,
        'pressure_msl': Weather.PRESSURE_SEA,
        'wind_gust_speed': Weather.WIND_SPEED_GUST,
        'wind_gust_direction': Weather.WIND_DIRECTION_GUST
    }, inplace=True)

    hours = pd.Series(data=data.index, index=data.index).diff().bfill().dt.total_seconds() / 3600.

    # Convert global horizontal irradiance from kWh/m^2 to W/m^2
    data[Weather.GHI] = data[Weather.GHI]*hours*1000

    data = data[[Weather.GHI,
                 Weather.TEMP_AIR,
                 Weather.TEMP_DEW_POINT,
                 Weather.HUMIDITY_REL,
                 Weather.PRESSURE_SEA,
                 Weather.WIND_SPEED,
                 Weather.WIND_SPEED_GUST,
                 Weather.WIND_DIRECTION,
                 Weather.WIND_DIRECTION_GUST,
                 Weather.CLOUD_COVER,
                 Weather.SUNSHINE,
                 Weather.VISIBILITY,
                 Weather.PRECIPITATION,
                 Weather.PRECIPITATION_PROB]]

    data.dropna(how='all', axis='columns', inplace=True)

    # Upsample forecast to a resolution of 1 minute. Use the advanced Akima interpolator for best results
    data = data.resample('1Min').interpolate(method='akima')

    # Drop interpolated irradiance values below 0
    data.loc[data[Weather.GHI] < 1e-3, Weather.GHI] = 0

    return data[start:end]
