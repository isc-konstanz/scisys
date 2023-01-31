"""
Open Power System Data

Household Datapackage

validation.py : fix possible errors and wrongly measured data.

"""
import logging
import os
import yaml
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
from datetime import datetime, timedelta
from th_e_core.tools import derive_power

logger = logging.getLogger(__name__)


def plot(feed: pd.DataFrame, feed_name, plot_dir: str = '/var/opt/th-e-data/plots', days: int = 7) -> None:
    """
    Plot energy and power values to visually validate data series

    Parameters
    ----------
    feed: pd.DataFrame
        DataFrame to inspect and possibly fix measurement errors
    feed_name : str
        Subset of feed columns available for the Household
    feed_name : str
        Name of the Data to indicate progress
    plot_dir : str
        Filepath of Plots to save created Graphs
    days: int
        Number of Days displayed on the created plots

    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.dates import DateFormatter
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()
    fig_counter = 0

    path = os.path.join(plot_dir, feed_name)
    if not os.path.exists(path):
        os.mkdir(path)

    plt.close('all')
    plt.rcParams.update({'figure.max_open_warning': 0})
    colors = plt.get_cmap('tab20c').colors

    week_start = feed.index[0]
    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
    week_start -= timedelta(days=week_start.dayofweek)
    while week_start <= feed.index[-1]:
        week_end = week_start + timedelta(days=days)
        week_data = feed[(feed.index >= week_start) & (feed.index < week_end)]

        errors = any('_error_' in column for column in week_data.columns) and \
                 not week_data.filter(regex="_error_").dropna(how='all').empty and \
                 any(week_data.filter(regex="_error_").dropna(how='all'))

        if week_data.empty or not errors:
            if week_data.empty:
                logger.warning('Skipping empty interval at index %s for %s', week_start.strftime('%d.%m.%Y %H:%M'),
                               feed_name)
            else:
                logger.debug('Skipping visualization for expected behaviour at index %s ',
                             week_start.strftime('%d.%m.%Y %H:%M'))

            week_start = week_end
            continue

        elif errors:
            fig, ax = plt.subplots(nrows=2)
            fig.autofmt_xdate()
            legend = []
            legend_names = []
            colors_counter = 0

            # for feed_name in feeds_name:
            feed_power = feed[feed_name.replace('energy', 'power')].dropna()
            feed = week_data.filter(regex=feed_name).dropna(how='all')
            feed.index = feed.index.tz_convert('Europe/Berlin')


            if feed_power.empty:
                continue

            ax[0].plot(feed_power.index, feed_power, color=colors[colors_counter], label=feed_name)
            ax[0].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax[0].set_ylabel('Power')

            energy = feed[feed_name] - feed.loc[feed.index[0], feed_name]
            ax[1].plot(feed.index, energy, color=colors[colors_counter])  # , linestyle='--')
            ax[1].xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))
            ax[1].set_ylabel('Energy')

            if errors:
                def plot_error(index, error_name, marker):
                    ax[index].plot(feed.index, feed[feed_name + error_name].replace(False, np.NaN).replace(True, -1),
                                   color=colors[colors_counter + 6],
                                   marker=marker,
                                   linestyle='None')

                plot_error(1, '_error_std', marker='x')
                plot_error(1, '_error_inc', marker='x')
                plot_error(0, '_error_qnt', marker='x')

            if any('_interpolated' in column for column in week_data.columns):
                ax[1].plot(feed.index, feed[feed_name + '_interpolated'].replace(True, -1),
                           color=colors[colors_counter], marker='o', linestyle='None')

            legend.append(Line2D([0], [0], color=colors[colors_counter]))
            legend_names.append(feed_name)

            colors_counter += 2
            if colors_counter == 20:
                colors_counter = 1

            ax[0].legend(legend, legend_names, loc='best')
            ax[0].title.set_text(feed_name)

            # save plots
            save_plot = os.path.join(plot_dir, feed_name, feed_name + '_' + str(fig_counter))
            pyplot.savefig(save_plot)
            fig_counter += 1

        week_start = week_end


# writes the found errors into the file failures_found
def create_fail_file(nan_blocks, name):
    errors = {str(name): []}
    adjustments = read_fail_file()

    if adjustments is None or name not in adjustments:
        if len(nan_blocks) > 0:
            for i in range(len(nan_blocks)):
                errors[name].append(
                    {
                        'type': 'remove',
                        'start': str(nan_blocks.get('start_idx')[i]),
                        'end': str(nan_blocks.get('till_idx')[i]),
                        'count': str(nan_blocks.get('count')[i]),
                    })
    else:
        adjustments_frame = pd.DataFrame.from_dict(adjustments[name])


        if len(nan_blocks) > 0:
            for i in range(len(nan_blocks)):
                if nan_blocks['start_idx'][i] not in adjustments_frame['start'] and \
                        nan_blocks['till_idx'][i] not in adjustments_frame['end']:
                    errors[name].append(
                        {
                            'type': 'remove',
                            'start': str(nan_blocks.get('start_idx')[i]),
                            'end': str(nan_blocks.get('till_idx')[i]),
                            'count': str(nan_blocks.get('count')[i]),
                        })
    return errors


def write_fail_file(errors):
    if errors is not None and bool(errors):
        with open('failures_found.yml', "a") as file:
            yaml.dump(errors, file)


def delete_fail_file():
    with open('failures_found.yml', 'w'):
        pass


def read_fail_file():
    with open('failures_found.yml', 'r+') as f:
        adjustments = yaml.load(f.read(), Loader=yaml.FullLoader)
    return adjustments


def correct(feed, feed_name):
    adjustments = read_fail_file()
    for items in adjustments[feed_name]:
        # read failures from file
        feed = series_adjustment(items, feed, feed_name)
    return feed


def write_csv(feed_frame, feed_name):
    feed_frame.drop(feed_name + '_error_std', axis=1, inplace=True)
    feed_frame.drop(feed_name + '_error_qnt', axis=1, inplace=True)
    feed_frame.drop(feed_name + '_error_inc', axis=1, inplace=True)

    start = feed_frame.index[0]

    while start <= feed_frame.index[-1]:
        end = start + timedelta(days=365)
        if end >= feed_frame.index[-1]:
            data = feed_frame[start:]
        else:
            while end not in feed_frame.index[:]:
                end = end + timedelta(seconds=1)
            data = feed_frame[start:end]
        data.to_csv(f'feed_fixed_{feed_name}_{str(start.year)}.csv')
        start = end


def find(feed, feed_name, unit, plot_dir, plot_draw=False, override=False):
    adjustments = read_fail_file()
    # feed does not exist in fail_file
    if adjustments is None or feed_name not in adjustments:
        errors, feed_frame = find_failures(feed, feed_name, unit, plot_dir)
        if errors is not None:
            write_fail_file(errors)
            feed = correct(feed, feed_name)
        errors, feed_frame = find_failures(feed, feed_name, unit, plot_dir=plot_dir, plot_data=plot_draw)

        adjustments = read_fail_file()
        adjustments.pop(feed_name)
        delete_fail_file()
        write_fail_file(adjustments)
        write_fail_file(errors)
    else:
        feed = correct(feed, feed_name)
        errors, feed_frame = find_failures(feed, feed_name, unit, plot_dir, plot_data=plot_draw)

        if override:
            adjustments.pop(feed_name)
            delete_fail_file()
            write_fail_file(adjustments)
            write_fail_file(errors)
        else:
            adjustments.update(errors)
            delete_fail_file()
            write_fail_file(adjustments)
    return feed_frame


def find_failures(feed, feed_name, unit, plot_dir, plot_data=False):
    feed_frame = feed.to_frame()

    # Find the rows where the energy values are not within +3 to -3 times the standard deviation.
    error_std, nan_std_blocks = find_standard_deviation(feed)

    nan_blocks = nan_std_blocks
    feed_frame[feed_name + '_error_std'] = error_std
    # Find the rows where the energy values is decreasing
    if unit == 'kWh':
        error_inc, nan_inc_blocks = find_energy_increasing_failure(feed)
        nan_blocks = pd.concat([nan_inc_blocks, nan_blocks], ignore_index=True)
        feed_frame[feed_name + '_error_inc'] = error_inc

    data_power = derive_power(feed).squeeze()
    feed_frame.insert(1, data_power.name, data_power)
    # Notify about rows where the derived power is significantly larger than the standard deviation value

    if unit == 'W' or not data_power.empty:
        error_qnt, nan_qnt_blocks = find_quantile_error_power(data_power)
        nan_blocks = pd.concat([nan_qnt_blocks, nan_blocks], ignore_index=True)
        feed_frame[feed_name + '_error_qnt'] = error_qnt

    count = []
    i = 0
    while i < len(nan_blocks):
        count.append(len(feed[nan_blocks['start_idx'][i]: nan_blocks['till_idx'][i]]))
        i += 1

    nan_blocks['count'] = count
    errors = create_fail_file(nan_blocks, feed_name)
    if plot_data:
        plot(feed_frame, str(feed_name), plot_dir=plot_dir)

    return errors, feed_frame


def find_quantile_error_power(feed):
    # Notify about rows where the derived power is significantly larger than the standard deviation value
    nan_blocks = pd.DataFrame()
    quantile = feed[feed > 0].quantile(.99)
    error_qnt = (feed.abs() > 3 * quantile)

    # first row of consecutive region is a True preceded by a False in tags
    nan_blocks['start_idx'] = feed.index[error_qnt & ~error_qnt.shift(1).fillna(False)]

    # last row of consecutive region is a False preceded by a True
    nan_blocks['till_idx'] = feed.index[error_qnt & ~error_qnt.shift(-1).fillna(False)]

    return error_qnt, nan_blocks


def find_energy_increasing_failure(feed):
    error_inc = feed < feed.shift(1)
    nan_blocks = pd.DataFrame()
    feed_size = len(feed.index)

    for time in error_inc.replace(False, np.NaN).dropna().index:
        # index of element i from error_inc
        i = feed.index.get_loc(time)
        if 2 <= i < feed_size:
            error_flag = None

            # If a rounding or transmission error results in a single value being too big,
            # fix that single data point, else flag all decreasing values
            if feed.iloc[i] >= feed.iloc[i - 2] and not error_inc.iloc[i - 2]:
                error_inc.iloc[i] = False
                error_inc.iloc[i - 1] = True

            elif all(feed.iloc[i - 1] > feed.iloc[i:min(feed_size - 1, i + 10)]):
                error_flag = feed.index[i]

            else:
                j = i + 1
                while j < feed_size and feed.iloc[i - 1] > feed.iloc[j]:
                    if error_flag is None and j - i > 10:
                        error_flag = feed.index[i]

                    error_inc.iloc[j] = True
                    j = j + 1

            if error_flag is not None:
                logger.warning('Unusual behaviour at index %s for  %s', error_flag.strftime('%d.%m.%Y %H:%M'),
                               feed.any())

    # first row of consecutive region is a True preceded by a False in tags
    nan_blocks['start_idx'] = feed.index[error_inc & ~error_inc.shift(1).fillna(False)]

    # last row of consecutive region is a False preceded by a True
    nan_blocks['till_idx'] = feed.index[error_inc & ~error_inc.shift(-1).fillna(False)]

    return error_inc, nan_blocks


def find_standard_deviation(feed):
    error_std = np.abs(feed - feed.mean()) > 3 * feed.std()

    # make another DF to hold info about each region
    nan_blocks = pd.DataFrame()

    # first row of consecutive region is a True preceded by a False in tags
    nan_blocks['start_idx'] = feed.index[error_std & ~error_std.shift(1).fillna(False)]

    # last row of consecutive region is a False preceded by a True
    nan_blocks['till_idx'] = feed.index[error_std & ~error_std.shift(-1).fillna(False)]

    return error_std, nan_blocks


def series_adjustment(adjustment, feed, feed_name):
    if 'start' in adjustment:

        # delete timezone information
        start_without_tz = adjustment['start'].split('+')[0]

        adj_start = pytz.timezone('UTC').localize(datetime.strptime(start_without_tz, '%Y-%m-%d %H:%M:%S'))
        if adj_start not in feed.index:
            logger.warning("Skipping adjustment outside index for in %s at %s", feed_name, adj_start)
            return feed
    else:
        adj_start = feed.index[0]

    if 'end' in adjustment:
        end_without_tz = adjustment['end'].split('+')[0]
        adj_end = pytz.timezone('UTC').localize(datetime.strptime(end_without_tz, '%Y-%m-%d %H:%M:%S'))
    else:
        adj_end = feed.index[-1]

    # TODO implement other types (remove, fill, etc)
    adj_type = adjustment['type']
    if adj_type == 'remove':
        # A whole time period needs to be removed due to very unstable transmission
        feed = feed.loc[(feed.index < adj_start) | (feed.index > adj_end)]

    elif adj_type == 'difference':
        # Changed smart meters, resulting in a lower counter value
        adj_index = feed.index.get_loc(adj_start)
        adj_delta = feed.iloc[adj_index - 1] - feed.iloc[adj_index]
        feed.loc[adj_start:adj_end] = feed.loc[adj_start:adj_end] + adj_delta

    logger.debug("Adjusted %s values (%s) from %s to %s", feed_name, adj_type, adj_start, adj_end)

    return feed


def validate(household: dict,
             household_data: pd.Series,
             image_dir: str = 'plot',
             draw_plot: bool = True,
             override: bool = False) -> pd.DataFrame:
    """
    Search for measurement faults in DataSeries and remove them

    Parameters
    ----------
    household : dict
        Configuration dictionary of the household
    household_data : pd.DataFrame
        Data to inspect and possibly fix measurement errors
    image_dir : str
        Directory path where all plots can be found
    draw_plot : boolean
        Flag, if the validated feeds should be printed as png file
    override : boolean
        Flag, if the already found failures should be deleted from the fail_file

    Returns
    ----------
    result: pandas.DataFrame
        Adjusted DataFrame with result series

    """
    logger.info('Validate %s series', household_data.name)

    feed_name = household_data.name
    unit = household[feed_name]['unit']
    feed = household_data.dropna()

    feed_frame = find(feed=feed,
                      feed_name=feed_name, unit=unit,
                      plot_dir=image_dir, plot_draw=draw_plot, override=override)

    feed_correct = correct(feed=feed,
                           feed_name=feed_name)

    feed_frame.drop(feed_name, axis=1, inplace=True)
    feed_frame[feed_name] = feed_correct
    write_csv(feed_frame, feed_name)
