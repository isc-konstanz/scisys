# -*- coding: utf-8 -*-
"""
    scisys.progress
    ~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Optional, Dict, Any

import json
import logging
import multiprocessing as process

logger = logging.getLogger(__name__)


class Progress:

    _instance = None

    @staticmethod
    def instance(*args, **kwargs):
        if Progress._instance is None:
            try:
                Progress._instance = Progress(*args, **kwargs)

            except ImportError as e:
                logger.debug("Unable to import tqdm progress library: %s", e)
                return None

        return Progress._instance

    @staticmethod
    def reset():
        Progress._instance = None

    @staticmethod
    def close():
        Progress.instance()._bar.close()

    # noinspection PyUnresolvedReferences
    def __init__(self,
                 desc: Optional[str] = None,
                 total: Optional[int] = None,
                 value: Optional[process.Value | int] = None,
                 file: Optional[str] = None,
                 **kwargs) -> None:
        from tqdm import tqdm

        # Manually disable the tqdm progress bars for testing on broken windows consoles
        # kwargs['disable'] = True
        # kwargs['file'] = sys.stdout

        if value is not None and type(value).__name__ == 'Synchronized' and value.value > 0:
            kwargs['initial'] = value.value

        self._total = total
        self._value = value
        self._file = file
        self._bar = tqdm(desc=desc, total=total, **kwargs)

    def complete(self, results: Optional[Dict[str, Any]] = None):
        if results is None:
            self._update(self._total, dump=True)
        else:
            self._update(self._total, dump=False)
            if self._file is not None:
                with open(self._file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

    # noinspection PyUnresolvedReferences
    def update(self):
        if self._value is not None:
            if type(self._value).__name__ == 'Synchronized':
                with self._value.get_lock():
                    self._value.value += 1
                    self._update(self._value.value)
            else:
                self._value += 1
                self._update(self._value)
        else:
            self._bar.update()

    def _update(self, value: int, dump: bool = True):
        progress = value/self._total*100
        if progress % 1 <= 1/self._total*100 and self._file is not None and dump:
            with open(self._file, 'w', encoding='utf-8') as f:
                # results = json.load(f)
                # if results['status'] != 'running':
                #     results['status'] = 'running'
                # results['progress'] = int(progress)
                results = {
                    'status': 'running',
                    'progress': int(progress)
                }
                json.dump(results, f, ensure_ascii=False, indent=4)

        self._bar.update(value - self._bar.n)

