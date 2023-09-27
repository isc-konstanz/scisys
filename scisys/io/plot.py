# -*- coding: utf-8 -*-
"""
    scisys.io.plot
    ~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import List

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logger = logging.getLogger(__name__)

show = False

COLORS = [
    '#004F9E',
    '#FFB800'
]

INCH = 2.54
WIDTH = 32/INCH
HEIGHT = 9/INCH


# noinspection PyDefaultArgument, SpellCheckingInspection
def line(x: pd.Series | str,
         y: pd.DataFrame | pd.Series | str,
         data: pd.DataFrame = None,
         title: str = '',
         xlabel: str = '',
         ylabel: str = 'Error [W]',
         color: List[str] = COLORS,
         file: str = None,
         **kwargs) -> None:

    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)

    color_num = max(len(data.columns) - 1, 1)
    colors = sns.color_palette(color, n_colors=color_num)
    plot = sns.lineplot(x=x,
                        y=y,
                        data=data,
                        palette=colors,
                        errorbar='sd',  # err_style="band", estimator=np.median,
                        **kwargs)

    if isinstance(x, str) and (x == 'hour' or x == 'horizon'):
        index_unique = data[x].astype(int).unique()
        index_unique.sort()
        plot.set_xticks(index_unique, labels=index_unique)

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)

    if show:
        plot.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


# noinspection PyDefaultArgument, SpellCheckingInspection
def bars(x: pd.Series | str,
         y: pd.DataFrame | pd.Series | str,
         data: pd.DataFrame = None,
         title: str = '',
         xlabel: str = '',
         ylabel: str = 'Error [W]',
         color: List[str] = COLORS,
         file: str = None,
         **kwargs) -> None:

    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)

    color_num = max(len(data.columns) - 1, 1)
    colors = sns.color_palette(color, n_colors=color_num)
    plot = sns.barplot(x=x,
                       y=y,
                       data=data,
                       palette=colors,
                       errorbar='sd',  # ci='sd',
                       **kwargs)

    if (not isinstance(x, str) and len(np.unique(x) > 24)) or len(data[x]) > 24:
        plot.xaxis.set_tick_params(rotation=45)

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)

    if show:
        plot.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


# noinspection PyDefaultArgument, SpellCheckingInspection
def quartiles(x: pd.Series | str,
              y: pd.DataFrame | pd.Series | str,
              data: pd.DataFrame = None,
              title: str = '',
              xlabel: str = '',
              ylabel: str = 'Error [W]',
              color: List[str] = COLORS,
              method: str = 'bar',
              file: str = None,
              **kwargs) -> None:

    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)

    color_num = max(len(data.columns) - 1, 1)
    colors = sns.color_palette(color, n_colors=color_num)

    if method in ['bar', 'bars']:
        fliers = dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='lightgrey')
        plot = sns.boxplot(x=x,
                           y=y,
                           data=data,
                           palette=colors,
                           flierprops=fliers,
                           # showfliers=False,
                           **kwargs)

        if (not isinstance(x, str) and len(np.unique(x) > 24)) or len(data[x]) > 24:
            plot.xaxis.set_tick_params(rotation=45)

    elif method == 'line':
        stats = data.groupby([x]).describe()
        index_values = stats.index
        index_unique = index_values.astype(int).unique().values
        index_unique.sort()

        medians = stats[(y, '50%')]
        quartile1 = stats[(y, '25%')]
        quartile3 = stats[(y, '75%')]

        plot = sns.lineplot(x=index_values,
                            y=medians,
                            color=colors[0])
        plot.fill_between(index_values, quartile1, quartile3, alpha=0.3)
        plot.set_xticks(index_unique, labels=index_unique)
    else:
        logger.error(f'Invalid boxplot method "{method}"')
        return

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)

    if show:
        plot.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


def histograms(data: pd.DataFrame,
               bins: int = 100,
               path: str = '') -> None:

    for column in data.columns:
        plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)

        # Create equal space bin values per column
        bin_data = []
        bin_domain = data[column].max() - data[column].min()
        bin_step = bin_domain / bins

        counter = data[column].min()
        for i in range(bins):
            bin_data.append(counter)
            counter = counter + bin_step

        # Add the last value of the counter
        bin_data.append(counter)

        bin_values, bin_data, patches = plt.hist(data[column], bins=bin_data)
        count_range = max(bin_values) - min(bin_values)
        sorted_values = list(bin_values)
        sorted_values.sort(reverse=True)

        # Scale plots by stepping through sorted bin_data
        for i in range(len(sorted_values) - 1):
            if abs(sorted_values[i] - sorted_values[i + 1]) / count_range < 0.80:
                continue
            else:
                plt.ylim([0, sorted_values[i + 1] + 10])
                break

        # Save histogram to appropriate folder
        path_dist = os.path.join(path, 'dist')
        path_file = os.path.join(path_dist, '{}.png'.format(column))
        if not os.path.isdir(path_dist):
            os.makedirs(path_dist, exist_ok=True)

        plt.title(r'Histogram of '+column)
        plt.savefig(path_file)

        if show:
            plt.show()
        plt.close()
        plt.clf()
