# -*- coding: utf-8 -*-
"""
    scisys.evaluation
    ~~~~~~~~~~~~~~~~~


"""
from __future__ import annotations
from collections.abc import Sequence
from typing import Tuple, List, Iterator

import os
import re
import logging
import numpy as np
import pandas as pd

# noinspection PyProtectedMember
from corsys.io._var import rename
from corsys.tools import to_int
from corsys import Settings, Configurations
from scisys.io import excel, plot
from copy import deepcopy
from math import sqrt
from .results import Results

logger = logging.getLogger(__name__)

TARGETS = {
    'pv': 'Photovoltaics',
    'el': 'Electrical',
    'th': 'Thermal'
}


class Evaluations(Sequence):

    # noinspection PyShadowingBuiltins
    def __init__(self, evaluations: List[Evaluation], dir: str = 'data') -> None:
        super().__init__()
        self._evaluations = evaluations
        self._evaluation_dir = dir

    def __iter__(self) -> Iterator[Evaluation]:
        return iter(self._evaluations)

    def __len__(self) -> int:
        return len(self._evaluations)

    def __getitem__(self, index: int) -> Evaluation:
        return self._evaluations[index]

    def __call__(self, results: List[Results]) -> None:
        index = []
        duration_columns = []
        evaluation_columns = []
        evaluations = {}
        for result in results:
            result_evaluations = self._get_valid(result)
            if len(result_evaluations) < 1:
                continue
            if result.name not in index:
                index.append(result.name)
            for duration in result.durations:
                duration_column = ('Durations [min]', str(duration))
                if duration_column not in duration_columns:
                    duration_columns.append(duration_column)
            for evaluation in result_evaluations:
                evaluation_column = (evaluation.header, evaluation.name)
                if evaluation_column not in evaluation_columns:
                    evaluation_columns.append(evaluation_column)
                evaluations[evaluation.name] = pd.DataFrame()
        columns = duration_columns + [('Total', 'Weighted')]
        for column in list(dict.fromkeys([c for c, _ in evaluation_columns])):
            columns += [c for c in evaluation_columns if c[0] == column]
        summary = pd.DataFrame(index=index, columns=pd.MultiIndex.from_tuples(columns))

        for result in results:
            result_evaluations = self._get_valid(result)
            if len(result_evaluations) < 1:
                continue
            for duration in result.durations.keys():
                summary.loc[result.name, ('Durations [min]', duration)] = round(result.durations[duration], 2)

            kpi_total_weights = 0
            kpi_total_sum = 0
            for evaluation in result_evaluations:
                kpi, kpi_data = evaluation(result)
                kpi_data = kpi_data.to_frame()
                kpi_column_name = evaluation.header
                if len(results) > 1:
                    kpi_data.columns = pd.MultiIndex.from_tuples([(evaluation.header, result.system.name)])
                else:
                    kpi_data.columns = [kpi_column_name]
                kpi_data.index.name = rename(evaluation.group)
                evaluations[evaluation.name] = pd.concat([evaluations[evaluation.name], kpi_data], axis='columns')
                if kpi:
                    summary.loc[result.name, (evaluation.header, evaluation.name)] = kpi

                # TODO: Add meaningful debugging
                kpi_total_weights += evaluation.weight
                kpi_total_sum += kpi*evaluation.weight
            kpi_total = kpi_total_sum/(kpi_total_weights if kpi_total_weights > 0 else 1)

            summary.loc[result.name, ('Total', 'Weighted')] = kpi_total

        if summary.drop('Total',           axis='columns', level=0, errors='ignore')\
                  .drop('Durations [min]', axis='columns', level=0, errors='ignore').empty:
            return

        excel.write(summary, evaluations, self._evaluation_dir)

    def _get_valid(self, results: Results) -> List[Evaluation]:
        return [e for e in self._evaluations if e.is_valid(results)]


class Evaluation:

    @classmethod
    def read(cls, settings: Settings) -> Evaluations:
        interval = settings.getint(Configurations.GENERAL, 'interval', fallback=None)
        evaluations = []
        evaluation_configs = Configurations.from_configs(settings,
                                                         conf_file='evaluations.cfg',
                                                         conf_dir=settings.dirs.conf,
                                                         require=False)

        override_path = os.path.join(settings.dirs.data, 'evaluations.cfg')
        if os.path.isfile(override_path):
            evaluation_configs.read(override_path, encoding='utf-8')

        for evaluation in [s for s in evaluation_configs.sections() if s.lower() != 'general']:
            evaluation_args = dict(evaluation_configs[evaluation].items())
            evaluation_args['interval'] = interval
            evaluations += cls._read(evaluation, **evaluation_args)

        return Evaluations(evaluations, settings.dirs.data)

    @classmethod
    def _read(cls, name: str, target: str, group: str, **kwargs) -> List[Evaluation]:
        evaluations = []
        for evaluation_target in target.split(','):
            evaluations.append(cls(name, evaluation_target.strip(), group, **kwargs))
        return evaluations

    def __init__(self,
                 name: str,
                 target: str,
                 group: str,
                 group_bins: int = None,
                 condition: str = None,
                 summary: str = 'mbe',
                 metric: str = 'mbe',
                 weight: float = 1,
                 plot: str = None,
                 **kwargs) -> None:

        self.name = name
        self.target = target
        self.group = group
        self.group_bins = group_bins
        self.condition = condition
        self.weight = float(weight)
        self.summary = summary
        self.metric = metric
        self.plot = plot

        if 'interval' in kwargs and kwargs['interval'] is not None and to_int(kwargs['interval']) > 1:
            # Verify forecast horizon condition to be greater than execution interval
            horizon_min = kwargs['interval']/60
            for condition in self.condition:
                condition_parts = re.split(' |!=|==|>=|<=|<|>', condition)
                if condition_parts[0] == 'horizon' and int(condition_parts[-1]) < horizon_min:
                    condition_index = self._condition.index(condition)
                    condition = condition[:condition.rfind(condition_parts[-1])] + str(horizon_min)
                    self._condition[condition_index] = condition

                    logger.debug(f'Evaluation "{self.name}" horizon condition too small for {horizon_min}h interval')

    def __call__(self, results: Results) -> Tuple[float, pd.Series]:
        # Prepare the data to contain only necessary columns and rows
        data = self._select(results)

        if np.isinf(data).values.any():
            data_inf = data[np.isinf(data).any(axis='columns')]
            data.drop(data_inf.index, inplace=True)
            logger.warning(f"Results datastore contains {len(data_inf)} infinit values")
        if data.isna().values.any():
            data_nan = data[data.isna().any(axis='columns')]
            data.drop(data_nan.index, inplace=True)
            logger.warning(f"Results datastore contains {len(data_nan)} invalid values")

        evaluation = self._process(data)
        summary = self._summarize(evaluation)

        self._plot(results, data, evaluation)

        return summary, evaluation

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name
        self._id = re.sub('[^a-zA-Z0-9-_ ]+', '', name).replace(' ', '_').lower()

    @property
    def columns(self):
        columns = [self._target,
                   self._target+'_ref',
                   self._target+'_err',
                   self.group]
        for condition in self.condition:
            condition_column = re.split(' |!=|==|>=|<=|<|>', condition)[0]
            if condition_column not in columns:
                columns.append(condition_column)
        return columns

    @property
    def header(self) -> str:
        target = self._target.lower().replace('_power', '').replace('_energy', '')
        if target in TARGETS:
            return TARGETS[target]
        return target.title()

    @property
    def target(self):
        return self._target+'_err'

    @target.setter
    def target(self, target: str) -> None:
        # Ensure proper formatting of string (no special signs)
        if re.match('[^a-zA-Z0-9-_ ]+', target):
            raise ValueError('An improper target name was passed for evaluation {}'.format(target))

        self._target = target.strip()

    @property
    def group(self) -> str:
        group = self._group
        if group == 'histogram':
            group = self.target
        return group

    @group.setter
    def group(self, group: str):
        # Ensure proper formatting of string (no special signs)
        if re.match('[^a-zA-Z0-9-_ ]+', group):
            raise ValueError('An improper group name was passed for evaluation {}'.format(group))

        self._group = group.strip()

    @property
    def group_bins(self) -> int:
        return self._group_bins

    @group_bins.setter
    def group_bins(self, bins: int | str):
        if not bins:
            bins = None
        elif not isinstance(bins, int):
            bins = int(bins)
        self._group_bins = bins

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, condition: str) -> None:
        self._condition = list()
        if not condition:
            return
        for c in condition.split(','):
            # Ensure proper formatting of string (no special signs or spaces)
            # TODO: Ensure that name is still a valid variable in the future
            if re.match('[^a-zA-Z0-9-_><=., ]+', c) or len(re.split('!=|==|>=|<=|<|>', c)) > 2:
                raise ValueError('The condition {} in the evaluation {} contains unsupported characters or operations'
                                 .format(c, self.name))

            self._condition.append(c.strip())

    @property
    def summary(self):
        return self._summary

    @summary.setter
    def summary(self, summary) -> None:
        self._summary = list()
        # Ensure proper formatting of string (no special signs)
        if re.match('[^a-zA-Z0-9-_ ]+', summary):
            raise ValueError('An improper summary name was passed for evaluation {}'.format(summary))

        whitelist = ['sum',
                     'mean',
                     'mbe',
                     'mae',
                     'rmse',
                     'weight_by_group',
                     'weight_by_bins',
                     'fullest_bin']

        summary = summary.lower().strip()
        if summary not in whitelist:
            raise ValueError('Summary type {} not of available options: {}'.format(summary, whitelist))

        self._summary = summary

    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, metric: str) -> None:
        # Ensure proper formatting of string (no special signs)
        if re.match('[^a-zA-Z0-9-_ ]+', metric):
            raise ValueError('An improper metric name was passed for evaluation {}'.format(metric))

        whitelist = ['sum',
                     'mean',
                     'mbe',
                     'mae',
                     'rmse']

        metric = metric.lower().strip()
        if metric not in whitelist:
            raise ValueError('Metric type {} not of available options: {}'.format(metric, whitelist))

        # TODO: verify allowed metrics
        self._metric = metric

    @property
    def plot(self):
        return self._plots

    @plot.setter
    def plot(self, plots: str) -> None:
        self._plots = list()
        if not plots:
            return
        for p in plots.split(','):
            # Ensure proper formatting of string (no special signs)
            if re.match('[^a-zA-Z0-9-_ ]+', p):
                raise ValueError('An improper plot name was passed for evaluation {}'.format(p))

            whitelist = ['line', 'bar']
            p = p.lower().strip()
            if p not in whitelist:
                raise ValueError('Plot type {} not of available options: {}'.format(p, whitelist))

            self._plots.append(p)

    def _plot(self, results: Results, data: pd.DataFrame, evaluation: pd.DataFrame | pd.Series) -> None:
        if len(self.plot) < 1:
            return

        plot_name = '{0}_{1}'.format(self._target.replace('_power', '').lower(), self.id)
        plot_dir = os.path.join(results.system.configs.dirs.data, 'plots')
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        for plot_type in self.plot:
            plot_title = '{0} {1}'.format(self.header, self.name)
            plot_file = os.path.join(plot_dir, '{0}_{1}.png'.format(plot_name, plot_type))
            plot_args = {'xlabel': rename(self.group), 'title': plot_title, 'file': plot_file}
            if '_power' in self._target:
                plot_args['ylabel'] = 'Error [W]'
            elif '_energy' in self._target:
                plot_args['ylabel'] = 'Error [kWh]'

            if plot_type == 'line':
                plot_args['hue'] = 'Results'
                plot_args['style'] = 'Results'
                if '_power' in self._target:
                    plot_args['ylabel'] = 'Power [W]'
                elif '_energy' in self._target:
                    plot_args['ylabel'] = 'Energy [kWh]'

                plot_data = data.rename(columns={
                    self._target:        'Prediction',
                    self._target+'_ref': 'Reference'
                })
                plot_data = plot_data[[self.group, 'Prediction', 'Reference']]
                plot_melt = plot_data.melt(id_vars=self.group, var_name='Results')

                plot.line(self.group, 'value', plot_melt, **plot_args)

            elif plot_type == 'bar':
                # def divide_by_max(d):
                #     maximum = data.loc[d.index, f'{self._target}_ref'].max()
                #     return (d / maximum) * 100 if maximum > 0 else 0

                if self.group_bins and self.group_bins > 1 or self.metric == 'sum':
                    if self._group == 'histogram':
                        plot_args['ylabel'] = 'Occurrences'
                        if '_power' in self._target:
                            plot_args['xlabel'] = 'Error [W]'
                        elif '_energy' in self._target:
                            plot_args['xlabel'] = 'Error [kWh]'

                    plot_data = evaluation.to_frame()
                    plot_index = evaluation.index.values.round(2)
                    plot.bar(plot_index, self.target, plot_data, **plot_args)

                else:
                    del plot_args['file']
                    plot_file = os.path.join(plot_dir, plot_name+'_quartiles_{0}.png')
                    plot_data = data[[self.group, self.target]]

                    plot.quartiles(self.group, self.target, plot_data,
                                   method='bars', file=plot_file.format('bar'), showfliers=False, **plot_args)

                    if len(data.groupby([self.group])) > 24:
                        plot.quartiles(self.group, self.target, plot_data,
                                       method='line', file=plot_file.format('line'), **plot_args)

                    # plot_args['ylabel'] = 'Error [%]'
                    # plot_file = os.path.join(plot_dir, f'{plot_name}_quartiles_norm.png')
                    # plot_data.loc[:, self.target] = plot_data.groupby(self.group).transform(divide_by_max)
                    # io.print_boxplot(plot_data, self.group, self.target, plot_file, **plot_args)

    def _select(self, results: Results) -> pd.DataFrame:
        data = deepcopy(results.data)

        columns = self.columns
        if 'hour' in columns:
            data['hour'] = data.index.hour + data.index.minute/60
        if 'day_of_week' in columns:
            data['day_of_week'] = data.index.day_of_week
        if 'day_of_year' in columns:
            data['day_of_year'] = data.index.day_of_year
        if 'month' in columns:
            data['month'] = data.index.month
        if 'year' in columns:
            data['year'] = data.index.year

        if self._target+'_est' in data.columns:
            logger.warning(f"Results datastore containing deprecated target name {self._target+'_est'}."
                           " Please generate results again.")
            data.rename(columns={self._target: self._target+'_ref'}, inplace=True)
            data.rename(columns={self._target+'_est': self._target}, inplace=True)

        for condition in self.condition:
            data.query(condition, inplace=True)

        return data[columns]

    def _process(self, data: pd.DataFrame) -> pd.Series:
        data = deepcopy(data)
        if self.group_bins and self.group_bins > 1:
            data = self._process_bins(data)
        else:
            data = getattr(self, '_process_{method}'.format(method=self.metric))(data)

        data.index.name = self.group
        data.name = self.target
        return data

    # noinspection SpellCheckingInspection
    def _process_bins(self, data: pd.DataFrame) -> pd.Series:
        if not self.group_bins or self.group_bins <= 1:
            raise ValueError("Unable to build bins for invalid value: " + str(self.group_bins))

        if self.target == self.group:
            bin_max = data[self._target+'_ref'].quantile(.95)
            bin_data, bin_edges = np.histogram(data[self.target], bins=self.group_bins, range=(-bin_max, bin_max))
            bin_vals = 0.5*(bin_edges[1:]+bin_edges[:-1])
        else:
            bin_data = []
            bin_vals = []
            _, bin_edges = np.histogram(data[self.group], bins=self.group_bins)
            for bin_left, bin_right in [(bin_edges[i], bin_edges[i+1]) for i in range(self.group_bins)]:
                bin_step = data.loc[(data[self.group] > bin_left) & (data[self.group] <= bin_right), self.target]
                bin_data.append(getattr(self, f'_summarize_{self.metric}')(bin_step))
                bin_vals.append(0.5*(bin_left + bin_right))

        return pd.Series(index=bin_vals, data=bin_data, name=self.target)

    def _process_sum(self, data: pd.DataFrame) -> pd.Series:
        data = data.groupby(self.group).sum()
        return data[self.target]

    def _process_mean(self, data: pd.DataFrame) -> pd.Series:
        data = data.groupby(self.group).mean()
        return data[self.target]

    def _process_mbe(self, data: pd.DataFrame) -> pd.Series:
        return self._process_mean(data)

    def _process_mae(self, data: pd.DataFrame) -> pd.Series:
        data[self.target] = data[self.target].abs()
        data = data.groupby(self.group).mean()
        return data[self.target]

    def _process_rmse(self, data: pd.DataFrame) -> pd.Series:
        data[self.target] = (data[self.target] ** 2)
        data = data.groupby(self.group).mean()
        return data[self.target].apply(lambda d: sqrt(d))

    def _summarize(self, data: pd.Series) -> float:
        return getattr(self, f'_summarize_{self.summary}')(data)

    @staticmethod
    def _summarize_sum(data: pd.Series) -> float:
        return data.sum()

    @staticmethod
    def _summarize_mean(data: pd.Series) -> float:
        return data.mean()

    @staticmethod
    def _summarize_mbe(data: pd.Series) -> float:
        return data.mean()

    @staticmethod
    def _summarize_mae(data: pd.Series) -> float:
        return data.abs().mean()

    @staticmethod
    def _summarize_rmse(data: pd.Series) -> float:
        return sqrt((data ** 2).mean())

    @staticmethod
    def _summarize_weight_by_group(data: pd.Series) -> float:
        group_max = data.index.max()
        group_scaling = np.array([(group_max - i)/group_max for i in data.index])
        group_weight = group_scaling/group_scaling.sum()
        return sqrt(((data ** 2) * group_weight).sum())

    @staticmethod
    def _summarize_weight_by_bins(data: pd.Series) -> float:
        bins_weight = data/data.sum()
        return sqrt(((data.index ** 2) * bins_weight).sum())

    @staticmethod
    def _summarize_fullest_bin(data: pd.Series) -> float | pd.Index:
        return data.idxmax()

    def is_valid(self, results: Results) -> bool:
        if self.target not in results.data.columns:
            return False
        if self.group not in results.data.columns and self.group not in \
                ['hour',
                 'day_of_week',
                 'day_of_year',
                 'month',
                 'year']:
            return False
        return True
