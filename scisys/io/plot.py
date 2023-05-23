# -*- coding: utf-8 -*-
"""
    scisys.io.plot
    ~~~~~~~~~~~~~~


"""
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARN)
logging.getLogger('matplotlib').setLevel(logging.WARN)
logger = logging.getLogger(__name__)

INCH = 2.54
WIDTH = 32/INCH
HEIGHT = 9/INCH


def print_lineplot(data, index, column, file,
                   title='', xlabel='', ylabel='Error [W]', color='#004F9E', **kwargs) -> None:
    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)
    color_num = max(len(data.columns) - 1, 1)
    colors = sns.dark_palette(color, n_colors=color_num, reverse=True)
    plot = sns.lineplot(x=index, y=column, ci='sd',  # errorbar='sd',
                        data=data,
                        palette=colors,
                        **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig = plot.figure
    fig.savefig(file)
    plt.close(fig)


def print_barplot(data, index, column, file,
                  title='', xlabel='', ylabel='Error [W]', color='#004F9E', **kwargs) -> None:
    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)
    color_num = max(len(data.columns) - 1, 1)
    colors = sns.dark_palette(color, n_colors=color_num, reverse=True)
    plot = sns.barplot(x=index, y=column, ci='sd',  # errorbar='sd',
                       data=data,
                       palette=colors,
                       **kwargs)
    if len(np.unique(index)) > 24:
        plot.xaxis.set_tick_params(rotation=45)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig = plot.figure
    fig.savefig(file)
    plt.close(fig)


def print_boxplot(data, index, column, file,
                  title='', xlabel='', ylabel='Error [W]', color='#004F9E', **kwargs) -> None:
    plt.figure(figsize=[WIDTH, HEIGHT], dpi=120, tight_layout=True)
    fliers = dict(marker='o', markersize=3, markerfacecolor='none', markeredgecolor='lightgrey')
    color_num = max(len(data.columns) - 1, 1)
    colors = sns.light_palette(color, n_colors=color_num, reverse=True)
    plot = sns.boxplot(x=index, y=column,
                       data=data,
                       palette=colors,
                       flierprops=fliers,  # showfliers=False,
                       **kwargs)
    plot.set(xlabel=xlabel, ylabel=ylabel, title=title)
    fig = plot.figure
    fig.savefig(file)
    plt.close(fig)


def print_histograms(data, bins=100, path=''):
    for column in data.columns:  # create 100 equal space bin values per column.
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
        plt.close()
        plt.clf()
