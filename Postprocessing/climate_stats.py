__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import calendar


def moving_avg_from_csvs(csvs, gcms, window, spinup, function='boxcar', time_units='D'):

    dfs = {}

    for csv in csvs.iterkeys():
        # load csvs into Pandas dataframes; reduce to columns of interest
        df = pd.read_csv(csvs[csv], index_col='Date', parse_dates=True)

        try:
            df = df[gcms]
        except KeyError:
            pass

        # trim spinup time from data to plot
        t0 = df.index[0]
        tspinup = np.datetime64(t0 + pd.DateOffset(years=spinup))
        df = df[tspinup:]

        # resample at daily interval so that time gaps are filled with NaNs
        df = df.resample(time_units)

        # smooth each column with moving average
        df = pd.rolling_window(df, window, function, center='true')

        dfs[csv] = df

    return dfs


def annual_timeseries(csvs, gcms, spinup, stat, calc='mean', quantile=None):


    # Error in case no quantile value is given
    if stat == 'quantile' and not quantile:
        raise ValueError("stat = 'quantile' require that a float argument be entered for quantile, "
                         "e.g. quantile=0.1 for Q90 flow")

    dfs = {}

    for csv in csvs.iterkeys():
        # load csvs into Pandas dataframes; reduce to columns of interest
        df = pd.read_csv(csvs[csv], index_col='Date', parse_dates=True)

        try:
            df = df[gcms]
        except KeyError: # in case not all of the scenarios are present
            pass

        # trim spinup time from data to plot
        t0 = df.index[0]
        tspinup = np.datetime64(t0+pd.DateOffset(years=spinup))
        df = df[tspinup:]

        # replace any zeros with NaNs
        #df = df.replace({0:np.nan})

        # find any years with only one value (due to date convention)
        # reset values for those years to NaN
        an = df.groupby(lambda x: x.year)
        singles = [group for group in an.groups.keys() if len(an.get_group(group)) == 1]

        # calculate annual mean or quantiles
        if stat == 'mean_annual':
            an = df.groupby(lambda x: x.year).agg(calc)
        elif stat == 'quantile':
            an = df.groupby(lambda x: x.year).quantile(q=quantile, axis=0)

        # reset index from int to datetime (have to map to str first)
        an.index = pd.to_datetime(map(str, an.index)).values
        an = an.resample('AS')

        # blast years with only 1 daily value from above
        for s in singles:
            dstart, dend = '%s-01-01' %(s), '%s-12-31' %(s)
            an[dstart:dend] = None

        dfs[csv] = an

    return dfs


def period_stats(csvs, compare_periods, stat, baseline_period=np.array([]),
                 calc='mean', quantile=None, normalize_to_baseline=False):
    '''
    Aggregates data from dict of csv files (e.g. gcm-scenario combindations) for a model variable
        - groups data by month or by year and calculates period statistics
    # Average values for 20th century baseline are compared to future scenarios
    # Returns list columns concatenated by month (n dates to compare x 12 months; or n dates of annual averages to compare)
    # Each column consists of values produced by each GCM-scenario combination

    Inputs:
    csvs: dictionary {scenario name: filename} of csv files, one for each climate scenario

    compare_periods: np array of [tstart, tend] for each time period (box)

    baseline_period: np array of [tstart, tend] for time period to average for "baseline value"

    stat: string 'Mean Monthly', 'Mean Annual', 'quantile'

    quantile: float if stat = 'quantile', specify quantile to compute (e.g., for Q90 flow, enter 0.1)

    normalize_to_baseline : report y-axis values relative to baseline period (as percentages)
    '''
    # Error in case no quantile value is given
    if stat == 'quantile' and not quantile:
        raise ValueError("stat = 'quantile' require that a float argument be entered for quantile, "
                         "e.g. quantile=0.1 for Q90 flow")

    dfs = []

    # build list of Pandas dataframes, one for each csv.
    for csv in csvs.iterkeys():
        df = pd.read_csv(csvs[csv], index_col='Date', parse_dates=True)

        # Rename columns so they're unique.
        df.columns = ['{}_{}'.format(csv, c) for c in df.columns]

        dfs.append(df)

    # join together
    df = dfs[0].join(dfs[1:])
    del dfs

    # build list of dataframes with box columns, one for each period
    box_data = []
    i=0
    for per in compare_periods:
        i += 1

        # set limits for slicing data to time period
        tstart, tstop = np.datetime64('%s-01-01' %(per[0])), np.datetime64('%s-12-31' %(per[1]))

        if stat == 'mean_monthly': # returns 12 months (rows) x n GCM-scenarios (columns) DataFrame
            #dfg = df[tstart:tstop].groupby(lambda x: x.month).agg(calc) # aggregate data by group, using calc operation
            dfg = df[tstart:tstop].resample('M', how=calc) # dataframe of monthly values, using calc operation
            dfg = dfg.groupby(dfg.index.month).agg('mean') # mean of monthly values for each month

            # give columns unique names based on month and time period
            columns = ['{} {}'.format(calendar.month_name[i], '-'.join(map(str, per))) for i in dfg.index]

            dfg = dfg.T # transpose data so that there is one column for each month;
            dfg.columns = columns # assign column names
            dfg = dfg.dropna() # drop GCM-scenarios that contain NaNs (no-data)

        elif stat == 'mean_annual':
            dfg = df[tstart:tstop].groupby(lambda x: x.year).agg(calc) # returns n years in period x n GCM-scenarios DataFrame

            #dfg = dfg.stack() # stack data into vector containing means for each year in period and gcm-scenario combination
            dfg = dfg.mean(axis=0).dropna() # make vector of means for each gcm-scenario combination

        # population of quantiles from all scenario-gcm combinations;
        # do not exclude zero values from quantile computations
        elif stat == 'quantile':
            #dfg = df[tstart:tstop].replace({0: np.nan}).groupby(lambda x: x.year).quantile(q=quantile, axis=0)
            dfg = df[tstart:tstop].groupby(lambda x: x.year).quantile(q=quantile, axis=0)
            # (returns n years in period x n GCM-scenarios DataFrame)

            #dfg = dfg.stack()
            dfg = dfg.mean(axis=0).dropna() # returns mean across n years in period, for each scenario that is not nan

        #dfg['date'] = dfg.index
        #dfg['period'] = ['-'.join(map(str, per))] * len(dfg)
        #dfg = pd.melt(dfg, var_name='scen', id_vars=['date', 'period'])

        box_data.append(dfg)
    del dfg

    # concatenate data so that boxes/columns are paried by month
    if stat == 'mean_monthly':
        tmp = []
        # Arrange columns into pairs for comparison
        for c in range(len(box_data[0].columns)):
            for p in range(len(compare_periods)):
                tmp.append(box_data[p].iloc[:, c])

        df_all = pd.DataFrame(tmp).T

    # annual data doesn't need to be concatenated
    else:
        df_all = box_data

    # now calculate baseline value by grouping the baseline period by month or year
    # (and for year, averaging the annual values to get single value for period)
    if len(baseline_period) == 2: # valid baseline period must have a start and end
        bstart, bstop = np.datetime64('%s-01-01' %(baseline_period[0])), np.datetime64('%s-12-31' %(baseline_period[1]))

        if stat == 'mean_monthly': # returns 12 (months) x 1 mean (of monthly means for all GCM-scenarios)
            bl = df[bstart:bstop].resample('M', how=calc)
            bl = bl.groupby(bl.index.month).agg('mean')
            bl = bl.mean(axis=1).values


        elif stat == 'mean_annual': # calculates annual mean values for each GCM, the mean for all GCMs (1 value/yr)
            # then takes the mean of all years in baseline period to get single value
            bl = df[bstart:bstop].groupby(lambda x: x.year).agg(calc)
            bl = np.array([bl.mean(axis=1).mean()])


        # population of quantiles from all scenario-gcm combinations; excludes 0 values
        elif stat == 'quantile':
            #bl = df[bstart:bstop].replace({0: np.nan}).groupby(lambda x: x.year).quantile(q=quantile, axis=0)
            bl = df[bstart:bstop].groupby(lambda x: x.year).quantile(q=quantile, axis=0)
            bl = np.array([bl.mean(axis=1).mean()])

        # make baseline value for each box
        bl_columns = np.array([])
        for b in range(len(bl)):
            for p in range(len(compare_periods)):
                bl_columns = np.append(bl_columns, bl[b])

        if normalize_to_baseline and stat == 'mean_annual' or normalize_to_baseline and stat == 'quantile':
            df_all = np.array([d.values for d in df_all]).transpose()
            df_all = 100 * df_all / bl_columns[0] - 100
            bl_columns = [0, 0]

        return df_all, bl_columns

    # if no baseline period supplied, just return the box columns
    else:
        return df_all
