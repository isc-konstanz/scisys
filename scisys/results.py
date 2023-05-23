# -*- coding: utf-8 -*-
"""
    scisys.results
    ~~~~~~~~~~~~~~


"""
from __future__ import annotations
from collections.abc import MutableMapping

import os
import pandas as pd

import scisys.io as io
from corsys.io import DatabaseUnavailableException
from corsys import System
from copy import deepcopy
from .durations import Durations


class Results(MutableMapping):

    def __init__(self, system: System, verbose: bool = False) -> None:
        self.system = system
        system_dir = system.configs.dirs.data
        data_dir = os.path.join(system_dir, 'results')
        if not os.path.isdir(data_dir):
            os.makedirs(data_dir, exist_ok=True)

        self._datastore = pd.HDFStore(os.path.join(data_dir, 'results.h5'))
        try:
            self._database = deepcopy(system.database)
            self._database.dir = data_dir
            self._database.enabled = True

        except DatabaseUnavailableException:
            self._database = None

        self.data = pd.DataFrame()
        self.durations = Durations(system)

        self.verbose = verbose

    def __setitem__(self, key: str, data: pd.DataFrame) -> None:
        self.set(key, data, how='concat')

    def __getitem__(self, key: str) -> pd.DataFrame:
        return self.get(key)

    def __delitem__(self, key: str) -> None:
        del self._datastore[key]

    def __iter__(self):
        return iter(self._datastore)

    def __len__(self) -> int:
        return len(self._datastore)

    def __contains__(self, key: str) -> bool:
        return f"/{key}" in self._datastore

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._database is not None:
            self._database.close()
        self._datastore.close()
        self.durations.stop()
        if self.verbose:
            for results_err in [c for c in self.data.columns if c.endswith('_err')]:
                results_file = os.path.join('results',
                                            'results_' + results_err.replace('_err', '').replace('_power', ''))
                results_data = self.data.reset_index().drop_duplicates(subset='time', keep='last')\
                                        .set_index('time').sort_index()

                io.write_csv(self.system, results_data, results_file)

    def set(self, key: str, data: pd.DataFrame, how: str = None) -> None:
        data.to_hdf(self._datastore, f"/{key}")
        if self._database is not None and self.verbose:
            data_file = os.path.join(self._database.dir, f"{key}.csv")
            data_dir = os.path.dirname(data_file)
            if not os.path.isdir(data_dir):
                os.makedirs(data_dir, exist_ok=True)
            self._database.write(data, file=data_file, rename=False)

        if how is None:
            return
        elif how == 'concat':
            self.data = pd.concat([self.data, data], axis='index')
        elif how == 'combine':
            self.data = data.combine_first(self.data)
        else:
            raise ValueError(f"invalid how option: {how}")

    def load(self, key: str, how: str = 'concat') -> pd.DataFrame:
        data = self.get(key)
        if how == 'concat':
            self.data = pd.concat([self.data, data], axis='index')
        elif how == 'combine':
            self.data = data.combine_first(self.data)
        else:
            raise ValueError(f"invalid how option: {how}")
        return data

    # noinspection PyTypeChecker
    def get(self, key: str) -> pd.DataFrame:
        return self._datastore.get(f"/{key}")

    @property
    def id(self):
        return self.system.id

    @property
    def name(self):
        return self.system.name
