# -*- coding: utf-8 -*-
"""
    scisys.report
    ~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import List

import logging

from corsys import Configurations, Configurable, Settings
from scisys import Results

logger = logging.getLogger(__name__)


class Report(Configurable):

    @classmethod
    def read(cls, settings: Settings) -> Report:
        return cls(Configurations.from_configs(settings,
                                               conf_file='report.cfg',
                                               conf_dir=settings.dirs.conf,
                                               require=False))

    @property
    def enabled(self) -> bool:
        if not self.configs.enabled or not self.configs.has_section(Configurations.GENERAL):
            return False
        # TODO: Indicate missing configs
        return all(self.configs.has_option(Configurations.GENERAL, k) for k in ['title', 'project', 'author'])

    def __call__(self, results: List[Results]) -> None:
        if not self.enabled:
            return

        from scisys.io.pdf import PdfWriter
        for result in results:
            pdf = PdfWriter.read(result.system, self.configs,
                                 confidential=self.configs.get(Configurations.GENERAL, 'confidential', fallback=False))

            self.build(pdf, result)
            self.save(pdf)

    def build(self, pdf, results: Results) -> None:
        pdf.add_cover(self.configs.get(Configurations.GENERAL, 'title'),
                      self.configs.get(Configurations.GENERAL, 'project'),
                      self.configs.get(Configurations.GENERAL, 'author'))

        pdf.add_table_of_content()

    # noinspection PyMethodMayBeStatic
    def save(self, pdf) -> None:
        pdf.save()
