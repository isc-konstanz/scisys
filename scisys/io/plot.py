# -*- coding: utf-8 -*-
"""
    scisys.io.plot
    ~~~~~~~~~~~~~~


"""
from __future__ import annotations
from typing import Optional, List

import os
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logger = logging.getLogger(__name__)

COLORS = [
    '#004F9E',
    '#FFB800'
]

INCH = 2.54
WIDTH = 32
HEIGHT = 9


# noinspection PyDefaultArgument, SpellCheckingInspection
def line(x: Optional[pd.Series | str] = None,
         y: Optional[pd.DataFrame | pd.Series | str] = None,
         data: Optional[pd.DataFrame] = None,
         title: str = '',
         xlabel: str = '',
         ylabel: str = '',
         xlim: tuple = None,
         ylim: tuple = None,
         grids: str = None,  # Either 'x', 'y' or 'both'
         colors: List[str] = COLORS,
         palette: Optional[str] = None,
         hue: Optional[str] = None,
         width: int = WIDTH,
         height: int = HEIGHT,
         show: bool = False,
         file: str = None,
         **kwargs) -> None:

    plt.figure(figsize=[width/INCH, height/INCH], dpi=120, tight_layout=True)

    color_num = max(len(np.unique(data[hue])) if hue else len(data.columns) - 1, 1)
    if color_num > 1:
        if palette is None:
            palette = f"blend:{','.join(colors)}"
        kwargs['palette'] = sns.color_palette(palette, n_colors=color_num)
        kwargs['hue'] = hue
    else:
        kwargs['color'] = colors[0]

    plot = sns.lineplot(x=x,
                        y=y,
                        data=data,
                        estimator=np.median,
                        **kwargs)

    if isinstance(x, str) and x in ['hour', 'horizon']:
        index_unique = data[x].astype(int).unique()
        index_unique.sort()
        plt.xticks(index_unique, labels=index_unique)

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.box(on=False)

    if xlim is not None:
        plt.xlim(xlim)

    if ylim is not None:
        plt.ylim(ylim)

    if grids is not None:
        plt.grid(color='grey', linestyle='--', linewidth=0.25, alpha=0.5, axis=grids)

    if show:
        plt.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


# noinspection PyDefaultArgument, SpellCheckingInspection
def bar(x: Optional[pd.Index | pd.Series | str] = None,
        y: Optional[pd.DataFrame | pd.Series | str] = None,
        data: Optional[pd.DataFrame] = None,
        title: str = '',
        xlabel: str = '',
        ylabel: str = '',
        bar_label_type: str = None,  # 'edge' or 'center'
        colors: List[str] = COLORS,
        palette: Optional[str] = None,
        hue: Optional[str] = None,
        width: int = WIDTH,
        height: int = HEIGHT,
        show: bool = False,
        file: str = None,
        **kwargs) -> None:

    plt.figure(figsize=[width/INCH, height/INCH], dpi=120, tight_layout=True)

    color_num = max(len(np.unique(data[hue])) if hue else len(data.columns) - 1, 1)
    if color_num > 1:
        if palette is None:
            palette = f"blend:{','.join(colors)}"
        kwargs['palette'] = sns.color_palette(palette, n_colors=color_num)
        kwargs['hue'] = hue
    else:
        kwargs['color'] = colors[0]

    plot = sns.barplot(x=x, y=y, data=data, **kwargs)

    if (isinstance(x, str) and len(data[x]) > 24) or (isinstance(x, (pd.Index, pd.Series)) and len(np.unique(x)) > 24):
        plot.xaxis.set_tick_params(rotation=45)

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)

    plt.box(on=False)

    if bar_label_type is not None:
        plot.bar_label(plot.containers[0], label_type=bar_label_type)

    if show:
        plt.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


def heatmap(data: pd.DataFrame,
            x: str, x_value_steps: int, xlabel: str,
            y: str, y_value_steps: int, ylabel: str,
            heatbar_label: str,
            operator: str, operator_value: str = None,
            file: str = None,
            show: bool = False,
            **kwargs) -> None:

    x_value_list = list(np.arange(
        x_value_steps * round(data[x].min() / x_value_steps) + x_value_steps,
        x_value_steps * round(data[x].max() / x_value_steps) + x_value_steps + 1, x_value_steps))
    y_value_list = list(np.arange(
        y_value_steps * round(data[y].min() / y_value_steps) - y_value_steps,
        y_value_steps * round(data[y].max() / y_value_steps) + 1, y_value_steps))
    y_value_list.sort(reverse=True)

    x_array = []
    value_array = []
    y_data = data

    for y_value in y_value_list:
        x_data = y_data[y_data[y] >= y_value]
        for x_value in x_value_list:
            datapoints = x_data[x_data[x] <= x_value]

            if operator == "datapoints":
                x_array.append(len(datapoints))

            if operator == "mean":
                x_array.append(round(datapoints[operator_value].mean(), 2))

            if operator == "median":
                x_array.append(round(datapoints[operator_value].median(), 2))

            if operator == "min":
                x_array.append(round(datapoints[operator_value].min(), 2))

            if operator == "max":
                x_array.append(round(datapoints[operator_value].max(), 2))

            x_data = x_data[x_data[x] > x_value]
        y_data = data[data[y] < y_value]
        value_array.append(x_array)
        x_array = []

    value_array = np.array(value_array)

    plot, ax = plt.subplots(figsize=[len(x_value_list) + len(x_value_list) / 4, len(y_value_list)])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    im = ax.imshow(value_array, cmap="Reds")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **{}, orientation="horizontal", shrink=0.5, **kwargs)
    cbar.ax.set_xlabel(heatbar_label)

    # Show all ticks and label them with the respective list entries
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(np.arange(len(x_value_list)) + 0.5, labels=x_value_list)
    plt.yticks(np.arange(len(y_value_list)) + 0.5, labels=y_value_list)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(y_value_list)):
        for j in range(len(x_value_list)):
            ax.text(j, i, value_array[i, j], ha="center", va="center", color="black")

    plot.tight_layout()

    if show:
        plt.show()
    if file is not None:
        plt.savefig(file)

    plt.close(plot.figure)
    plt.clf()


# noinspection PyDefaultArgument, SpellCheckingInspection
def quartiles(x: Optional[pd.Series | str] = None,
              y: Optional[pd.DataFrame | pd.Series | str] = None,
              data: Optional[pd.DataFrame] = None,
              title: str = '',
              xlabel: str = '',
              ylabel: str = '',
              method: str = 'bar',
              colors: List[str] = COLORS,
              palette: Optional[str] = None,
              hue: Optional[str] = None,
              width: int = WIDTH,
              height: int = HEIGHT,
              show: bool = False,
              file: str = None,
              **kwargs) -> None:

    plt.figure(figsize=[width/INCH, height/INCH], dpi=120, tight_layout=True)

    color_num = max(len(np.unique(data[hue])) if hue else len(data.columns) - 1, 1)
    if color_num > 1:
        if palette is None:
            palette = f"blend:{','.join(colors)}"
        kwargs['palette'] = sns.color_palette(palette, n_colors=color_num)
        kwargs['hue'] = hue
    else:
        kwargs['color'] = colors[0]

    if method in ['bar', 'bars']:
        fliers = dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='lightgrey')
        plot = sns.boxplot(x=x,
                           y=y,
                           data=data,
                           flierprops=fliers,
                           **kwargs)

        if (isinstance(x, str) and len(data[x]) > 24) or (isinstance(x, pd.Series) and len(np.unique(x)) > 24):
            plot.xaxis.set_tick_params(rotation=45)

    elif method == 'line':
        # stats = data.groupby([x]).describe()
        # index_values = stats.index
        # index_unique = index_values.astype(int).unique().values
        # index_unique.sort()
        #
        # medians = stats[(y, '50%')]
        # quartile1 = stats[(y, '25%')]
        # quartile3 = stats[(y, '75%')]

        plot = sns.lineplot(x=x,
                            y=y,
                            data=data,
                            errorbar=('pi', 50),
                            estimator=np.median,
                            **kwargs)

        # plot.fill_between(index_values, quartile1, quartile3, color=color_palette[0], alpha=0.3)

        if isinstance(x, str) and x in ['hour', 'horizon']:
            index_unique = data[x].astype(int).unique()
            index_unique.sort()
            plot.set_xticks(index_unique, labels=index_unique)

    else:
        logger.error(f'Invalid boxplot method "{method}"')
        return

    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)

    if show:
        plt.show()
    if file is not None:
        plot.figure.savefig(file)

    plt.close(plot.figure)
    plt.clf()


def histograms(data: pd.DataFrame,
               bins: int = 100,
               show: bool = False,
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
