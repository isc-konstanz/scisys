# -*- coding: utf-8 -*-
"""
    scisys.durations
    ~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Iterator

import os
import json
import datetime as dt

from corsys import System


class Durations(Mapping):

    # noinspection PyUnresolvedReferences
    def __init__(self, system: System) -> None:
        self._durations = {}
        self._file = os.path.join(system.configs.dirs.data, 'results', 'results.json')
        if os.path.isfile(self._file):
            with open(self._file, 'r', encoding='utf-8') as f:
                self._durations = json.load(f)
                for duration in self._durations.values():

                    # noinspection PyUnresolvedReferences
                    def _datetime(key):
                        return dt.datetime.strptime(duration[key], '%Y-%m-%d %H:%M:%S.%f')

                    if 'start' in duration:
                        duration['start'] = _datetime('start')
                    if 'end' in duration:
                        duration['end'] = _datetime('end')

    def __repr__(self) -> str:
        return str(self._durations)

    def __iter__(self) -> Iterator:
        return iter(self._durations)

    def __len__(self) -> int:
        return len(self._durations)

    def __getitem__(self, key: str) -> float:
        return self._durations[key]['minutes']

    def start(self, key: str) -> None:
        if key not in self._durations:
            self._durations[key] = {}
        if 'minutes' not in self._durations[key]:
            self._durations[key]['minutes'] = 0
        if 'end' in self._durations[key]:
            del self._durations[key]['end']

        self._durations[key]['start'] = dt.datetime.now()

    def stop(self, key: str = None) -> None:
        if key is None:
            for key in self.keys():
                self._stop(key)
        else:
            self._stop(key)

        self._write()

    def _stop(self, key: str = None) -> None:
        if key not in self._durations:
            raise ValueError("No duration found for key: \"{}\"".format(key))
        if 'start' not in self._durations[key]:
            raise ValueError("Timer for key \"{}\" not started yet".format(key))

        self._durations[key]['end'] = dt.datetime.now()

        minutes = self._durations[key]['minutes'] if 'minutes' in self._durations[key] else 0
        minutes += round((self._durations[key]['end'] - self._durations[key]['start']).total_seconds() / 60.0, 6)
        self._durations[key]['minutes'] = minutes

    def _write(self) -> None:
        with open(self._file, 'w', encoding='utf-8') as f:
            json.encoder.FLOAT_REPR = lambda o: format(o, '.3f')
            json.dump(self._durations, f, indent=4, default=str, ensure_ascii=False)
