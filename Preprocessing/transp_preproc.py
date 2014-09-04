# generates *transp.day files for GSFLOW write climate module

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import PRMSio
import textwrap
import sys
sys.path.append('../Postprocessing')
import climate_plots as cp

# inputs
datadir = 'D:/ATLData/Fox-Wolf/data' # contains existing PRMS .data files

# growing season parameters
uniform = False # T/F; T: one growing season for entire domain (incomplete option), F: growing season by hru
nhru = 880
frost_temp = 28.0
growing_output = False # if True, generate .day files, otherwise just plots (much faster)
real_data_periods = ['1961-2000', '2046-2065', '2081-2100'] # for labeling non-synthetic data on plots

# output (.day files will be saved to datadir, with 'ide.data' endings replaced with 'transp.day')
outpdf = 'D:/ATLData/Fox-Wolf/Fox-Wolf_growing_season.pdf'

# growing season plot properties
props = {'sresa1b': {'color': 'Tomato', 'zorder': 2, 'alpha': 0.5},
         'sresa2': {'color': 'SteelBlue', 'zorder': 1, 'alpha': 0.5},
         'sresb1': {'color': 'Yellow', 'zorder': 3, 'alpha': 0.5}}

datafiles = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.data')]
scenarios = list(np.unique([f.split('.')[1] for f in datafiles]))

'''
print 'determining growing seasons...\n'
Dfg_days_starts = defaultdict()
Dfg_days_ends = defaultdict()

count = 0
'''

def uniform_growing_season(df, data, frost_temp):
    # initialize new dataframe to store growing season starts and ends
    # start with dummy values for year before simulation period
    year = int(data.timeper[0:4])-1
    dummyseason = np.array([[np.datetime64(dt.datetime(year, 1, 1)),
                                      np.datetime64(dt.datetime(year, 7, 1))]])
    dfg = pd.DataFrame(dummyseason, index=[year])
    dfg_days = pd.DataFrame(np.array([[1, 183]]), index=[pd.Timestamp(dt.datetime(year, 1, 1))],
                            columns=['Growing season start', 'Growing season end'])
    growing = False
    year = int(data.timeper[0:4])

    # iterate through data file to determine growing season starts and ends for each year
    print "finding last and first days with tmin<%s for each year..." %(frost_temp)
    for index, row in df.iterrows():
        yr = index.year

        # while in the same year
        if yr == year:
            Tmin = np.min(row)

            # if minimum temp rises above frost_temp, reset growing_start
            # stop reseting growing_start after July 1
            if not growing and Tmin > frost_temp:
                if index < pd.Timestamp(np.datetime64(dt.datetime(yr, 7, 1))):
                    growing_start = index
                    growing = True

            # if min temp falls back below, turn off growing so start will be eventually reset above
            # after July 1, once growing is turned off, it won't be turned on again
            if growing and Tmin < frost_temp:
                growing = False
                growing_end = index

        # once a new year has been reached
        # record growing season starts and ends from previous year; variables will reset themselves
        # also record starts/ends as days of year for plotting
        if yr <> year: #otherwise last year won't append
            dftemp = pd.DataFrame(np.array([[growing_start, growing_end]]), index=[year])
            dftemp_days = pd.DataFrame(np.array([[growing_start.dayofyear, growing_end.dayofyear]]),
                                       index=[pd.Timestamp(dt.datetime(year, 1, 1))],
                                       columns=['Growing season start','Growing season end'])
            dfg = dfg.append(dftemp)
            dfg_days = dfg_days.append(dftemp_days)

            year = yr

    # append last year
    dftemp = pd.DataFrame(np.array([[growing_start, growing_end]]), index=[year])
    dftemp_days = pd.DataFrame(np.array([[growing_start.dayofyear, growing_end.dayofyear]]),
                               index=[pd.Timestamp(dt.datetime(year, 1, 1))],
                               columns=['Growing season start', 'Growing season end'])
    dfg = dfg.append(dftemp)
    dfg_days = dfg_days.append(dftemp_days)

    # trim dummy year from start:
    dfg = dfg[1:]
    dfg_days = dfg_days[1:]

    return dfg, dfg_days


def growing_season_by_hru(df, frost_temp):
    '''
    find last date where tmin is < frost_temp
    '''
    print '\ndetermining growing season...'
    # group tmin data by year
    dfg = df.groupby(lambda x: x.year)

    # initialize dataframes for last and first frosts by hru
    years = [y for y, g in dfg]
    df_lf = pd.DataFrame(index=years, columns=df.columns)
    df_ff = pd.DataFrame(index=years, columns=df.columns)

    # iterate through years and dataframes in dfg
    for year, dfy in dfg:
        print year,
        # reindex dataframe to day of year
        dfy.index = np.arange(len(dfy))+1

        # split year in half on July 1st
        dfy1, dfy2 = dfy.iloc[:182, :], dfy.iloc[182:, :]

        # last frost is the last day before July 1st on which tmin <= frost_temp (returns single row vector)
        # first frost is the first day after July 1st on which tmin <= frost_temp
        last_frost = dfy1.apply(lambda x: x[x <= frost_temp].dropna().index[-1])
        first_frost = dfy2.apply(lambda x: x[x <= frost_temp].dropna().index[0])

        # assign (1 x nhru) vectors of last/first frosts to year in DataFrame
        df_lf.loc[year, :] = last_frost
        df_ff.loc[year, :] = first_frost
    print '\n'
    return df_lf, df_ff


##################
## Main Program ##
##################

# dict of dataframes, one per future emissions scenario
gsl = dict(zip(scenarios, [pd.DataFrame()] * len(scenarios)))

for f in datafiles:

    data = PRMSio.datafile(f)

    df = data.read2df()

    # slice dataframe to columns representing tmin data (all we need for determining growing season)
    df = df.loc[:, data.tmin_start: data.tmin_stop]
    df.columns = np.arange(np.shape(df)[1]) + 1 # hru numbers as column names

    if uniform:

        #This option needs some more work to mesh back with the original code (commented out) below!
        dfg, dfg_days = uniform_growing_season(df, data, frost_temp)

    else:
        # get dataframes of last and first frost (in days since Jan1) by hru (n years x n hrus)
        df_lf, df_ff = growing_season_by_hru(df, frost_temp)

        # munge data for growing season length summary plot
        # calculate average growing season length across all hrus
        gs_length = (df_ff-df_lf).mean(axis=1)

        # reset index to datetime to avoid confusion when adding to scenario dataframe (pandas won't match up integer indexes)
        gs_length.index = pd.DatetimeIndex([dt.datetime(y, 1, 1) for y in gs_length.index])

        # populate in dataframe for scenario (column name = gcm)
        gsl[data.scenario] = gsl[data.scenario].append(pd.DataFrame({data.gcm: gs_length}))

        if growing_output:
            # instantiate dotDay class
            dD = PRMSio.dotDay()

            # convert df_lf and df_ff to boolean dataframe of growing season (n days x n hrus)
            dD.transp(df_lf, df_ff, nhru)

            # write transp.day output file
            dD.header = ['created by transp_preproc.py\ntransp_on     %s\n' %(nhru) + 40*'#' + '\n']

            outfile = f[:-5] + '_transp.day'
            dD.write_output(outfile)

# Make a summary plot of growing season length for suite of GCMs and future emissions scenarios

# finish munging the data
if '20c3m' in scenarios:
    scenarios = [s for s in scenarios if s != '20c3m']
    for s in scenarios:
        gsl[s] = gsl[s].append(gsl['20c3m']).sort()

# resample the timeseries of growing season lengths at annual intervals (based on Jan1),
# so that no-data periods are filled in by nans
for s in scenarios:
    gsl[s] = gsl[s].resample('AS')

fig, ax = cp.timeseries(gsl, ylabel='Growing season length (days)', props=props, title='')
fig.savefig(outpdf, dpi=300)








'''
Old code that was used with BEC data to develop single growing season for whole domain

    # append to dicts for plotting
    Dfg_days_starts[f] = dfg_days['Growing season start']
    Dfg_days_ends[f] = dfg_days['Growing season end']
    count += 1
    
    if growing_output:
        
        DFG_days = pd.DataFrame()

        for index, row in dfg.iterrows():
            # fill in growing data for the year (days x nhru)
            Jan1, Dec31 = pd.Timestamp(datetime.datetime(index, 1, 1)), pd.Timestamp(datetime.datetime(index, 12, 31))
            index1 = pd.date_range(start=Jan1, end=row[0])
            index2 = pd.date_range(start=row[0], end=row[1])
            index3 = pd.date_range(start=row[1], end=Dec31)
            days = pd.DataFrame(data=np.zeros((len(index1), nhru), dtype=int), index=index1)
            days = days.append(pd.DataFrame(data=np.ones((len(index2), nhru), dtype=int), index=index2))
            days = days.append(pd.DataFrame(data=np.zeros((len(index3), nhru), dtype=int), index=index3))
            # append to rest
            DFG_days = DFG_days.append(days)
        
        fmt = '{:}'.format
        outfile = "%stransp.day" %(f[:-8])
        newheader = ['created by transp_preproc.py', 'transp_on     %s' %(nhru), 40*'#']
        PRMS_data_interp_functions.writeoutput(DFG_days, newheader, outfile, datadir, fmt)

    print '\n%s%% done\n' %(round(100.0*count/float(len(datafiles)),1))
        
        
print "plotting growing season results; saving to %s" %(outpdf)
OutPlot=PdfPages(outpdf)

# summary plot of results for all GCM-Scenarios
DFg_days_start=pd.DataFrame(Dfg_days_starts,index=dfg_days.index)
DFg_days_end=pd.DataFrame(Dfg_days_ends,index=dfg_days.index)

fig, (ax1)= plt.subplots(1,1)
colors=["r","b","y"]

# splitout columns into scenarios
i=0
p=defaultdict()
for s in scenarios:
    scen_cols=[c for c in DFg_days_start.columns if s in c]
    ax1.fill_between(DFg_days_start.index,DFg_days_end[scen_cols].max(axis=1),DFg_days_start[scen_cols].min(axis=1),alpha=0.3,color=colors[i])    
    p[s]=plt.Rectangle((0,0),1,1,fc=colors[i])    
    i+=1

plt.legend(p.values(), p.keys(), title='Emissions scenarios', loc='center right')
title = 'Simulated Growing Season - Max/Min values for all GCMs'
ax1.set_title(title)
ax1.set_ylabel('Day of Year')

# space out the x ticks so they aren't crammed together
xloc=plt.MaxNLocator(10)
ax1.xaxis.set_major_locator(xloc)
ax1.set_xlabel('Year')

plt.grid(True)

OutPlot.savefig()


# individual plots by GCM-Scenario
for f in datafiles:
    print "\t%stransp.day" %(f[:-8])
    fig, (ax1)= plt.subplots(1,1)
    # shade time periods containing "real" downscaled data
    for p in real_data_periods:
        start,end=map(int,p.split('-'))
        start,end=datetime.datetime(start,1,1),datetime.datetime(end,1,1)
        ndays=len(pd.date_range(start,end))
        ax1.fill_between(pd.date_range(start,end),np.ones(ndays)*350,np.ones(ndays)*50,color='0.75')
    
    
    ax1.fill_between(DFg_days_start.index,Dfg_days_ends[f],Dfg_days_starts[f])
    
    # create legend
    real_dps=plt.Rectangle((0,0),1,1,fc='0.5')
    gs=plt.Rectangle((0,0),1,1,fc='b')
    plt.legend([gs,real_dps],['Growing Season','Downscaled Wicci2\nData (white is synthetic)'],loc='center right')
    
    # wrap the title
    wrap=60
    title='Simulated Growing Season for %s' %(f)
    title="\n".join(textwrap.wrap(title,wrap))
    ax1.set_title(title)
    
    ax1.set_ylabel('Day of Year')
    
    # space out the x ticks so they aren't crammed together
    xloc=plt.MaxNLocator(10)
    ax1.xaxis.set_major_locator(xloc)
    
    ax1.set_xlabel('Year')
    plt.grid(True)
    OutPlot.savefig()
    
OutPlot.close()

print "Done!"

'''
