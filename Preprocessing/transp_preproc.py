# generates *transp.day files for GSFLOW write climate module

import os
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import PRMSio
import textwrap

# inputs
datadir = 'D:/ATLData/Fox-Wolf/data' # contains existing PRMS .data files

# growing season parameters
uniform = False # T/F; T: one growing season for entire domain, F: growing season by hru
nhru = 880
frost_temp = 28.0
growing_output = False # if True, generate .day files, otherwise just plots (much faster)
real_data_periods = ['1961-2000', '2046-2065', '2081-2100'] # for labeling non-synthetic data on plots

# output (.day files will be saved to datadir, with 'ide.data' endings replaced with 'transp.day')
outpdf = 'growing_season_plots.pdf'

datafiles = [os.path.join(datadir, f) for f in os.listdir(datadir) if f.endswith('.data')]
scenarios = list(np.unique([f.split('.')[1] for f in datafiles]))

print 'determining growing seasons...\n'
Dfg_days_starts = defaultdict()
Dfg_days_ends = defaultdict()

count = 0


def uniform_growing_season(df, timeper):
    # initialize new dataframe to store growing season starts and ends
    # start with dummy values for year before simulation period
    year = int(timeper[0:4])-1
    dummyseason = np.array([[np.datetime64(datetime.datetime(year, 1, 1)),
                                      np.datetime64(datetime.datetime(year, 7, 1))]])
    dfg = pd.DataFrame(dummyvalues, index=[year])
    dfg_days = pd.DataFrame(np.array([[1, 183]]), index=[pd.Timestamp(datetime.datetime(year, 1, 1))],
                            columns=['Growing season start', 'Growing season end'])
    growing = False
    year = int(timeper[0:4])

    # iterate through data file to determine growing season starts and ends for each year
    print "finding last and first days with tmin<%s for each year..." %(frost_temp)
    for index, row in df.iterrows():
        yr = index.year

        # while in the same year
        if yr == year:
            Tmin = np.min(row.values[tmax_cols:tmax_cols+tmin_cols])

            # if minimum temp rises above frost_temp, reset growing_start
            # stop reseting growing_start after July 1
            if not growing and Tmin > frost_temp:
                if index < pd.Timestamp(np.datetime64(datetime.datetime(yr, 7, 1))):
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
                                       index=[pd.Timestamp(datetime.datetime(year, 1, 1))],
                                       columns=['Growing season start','Growing season end'])
            dfg = dfg.append(dftemp)
            dfg_days = dfg_days.append(dftemp_days)

            year = yr

    # append last year
    dftemp = pd.DataFrame(np.array([[growing_start, growing_end]]), index=[year])
    dftemp_days = pd.DataFrame(np.array([[growing_start.dayofyear, growing_end.dayofyear]]),
                               index=[pd.Timestamp(datetime.datetime(year, 1, 1))],
                               columns=['Growing season start', 'Growing season end'])
    dfg = dfg.append(dftemp)
    dfg_days = dfg_days.append(dftemp_days)

    # trim dummy year from start:
    dfg = dfg[1:]
    dfg_days = dfg_days[1:]

    return dfg, dfg_days


def growing_season(dfy, frost_temp):
    '''
    find last date where tmin is < frost_temp
    takes a Dataframe of values from one year
    '''

    # reindex dataframe to day of year
    dfy.index = np.arange(len(dfy))+1

    # split year in half on July 1st
    dfy1, dfy2 = dfy.iloc[:182, :], dfy.iloc[182:, :]

    # last frost is the last day before July 1st on which tmin <= frost_temp (returns single row vector)
    last_frost = dfy1.apply(lambda x: x[x <= frost_temp].dropna().index[-1])
    first_frost = dfy2.apply(lambda x: x[x <= frost_temp].dropna().index[0])



for f in datafiles[0:1]:

    data = PRMSio.datafile(f)

    df = data.read2df()

    # slice dataframe to columns representing tmin data (all we need for determining growing season)
    df = df.loc[:, data.tmin_start: data.tmin_stop]
    df.columns = np.arange(np.shape(df)[1]) + 1 # hru numbers as column names


    '''
    if uniform:
        dfg, dfg_days = uniform_growing_season(df, data.timeper)


    
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
