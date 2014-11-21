# programs for aggregating and plotting results from GSFLOW climate change scenario runs

import os
import zipfile
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.dates as mdates
import matplotlib.cm as cm
from collections import defaultdict
import textwrap
import calendar
import pdb

def check4output_folder(mode):
    dirs=[f for f in os.listdir(os.getcwd()) if os.path.isdir(f)]
    if mode not in dirs:
        os.makedirs(mode)

def getvars(results_folder,mode):
    # results_folder: master folder with zipped results
    # mode: 'statvar','csvs','ggo','ssf'
    fromZip=False
    allfiles=os.listdir(results_folder)
    zipfiles=[f for f in allfiles if f.endswith('.zip')]
    
    if len(zipfiles)>0:
        fromZip=True
        zfile=zipfile.ZipFile(os.path.join(results_folder,zipfiles[0]))
    
    # exclude empty files
    fileslist=[]
    print '\nChecking for empty output files...'
    if fromZip:
        for item in zfile.infolist():
            if item.file_size>0:
                print item.filename
                fileslist.append(item.filename)
            else:
                print 'excluding %s, size=0' %(item.filename)
        varfile=[f for f in fileslist if mode.lower() in f.lower()][0]
    else:
        fileslist=[f for f in allfiles if f.endswith('.csv')]
    
    if mode=='statvar':
        if fromZip:
            statvardata=zfile.open(varfile).readlines()
            nvars=int(statvardata[0].strip())
            varlist=[]
            for n in range(nvars+1)[1:]:
                var=statvardata[n].strip().split()[0]
                varlist.append(var)
        else:
            varlist=list(set([f.split('.')[0] for f in fileslist])) # set() returns unique items
            
    elif mode=='csv':
        if fromZip:
            csvdata=zfile.open(varfile).readline()
            varlist=csvdata.strip().split(',')[1:]
        else:
            varlist=list(set([f.split('.')[0] for f in fileslist]))
            
    elif mode=='ggo':
        if fromZip:
            varlist=[f[:-4] for f in fileslist if f.endswith(mode)]
        else:
            varlist=list(set([f.split('.')[0] for f in fileslist]))
        varlist=[f for f in varlist if not 'uzf' in f.lower()]
        
    elif mode=='ssf':
        if fromZip:
            ssfdata=zfile.open(varfile).readlines()
            varlist=[]
            for line in ssfdata:
                varline=line.strip().split()[0]
                if varline not in varlist:
                    varlist.append(varline)
        else:
            varlist=list(set([f.split('.')[0] for f in fileslist]))
            
    elif mode.lower()=='uzf':
        if fromZip:
            uzfdata=zfile.open(varfile).readlines()[1]
            varlist=uzfdata.strip().split()[2:]
            varlist=[f for f in varlist if f<>'"'] # get rid of double quotes
        else:
            varlist=list(set([f.split('.')[0] for f in fileslist]))
            
    print '\nVariables to aggregate:'
    # this caused a problem with ggo file bec_lf0.5.ggo
    #varlist=[var.replace('.','') for var in varlist] # get rid of any dots in variable names
    for var in varlist:
        print var
    return varlist


def getscenarios(results_folder):
    fromZip=False
    allfiles=os.listdir(results_folder)
    zipfiles=[f for f in allfiles if f.endswith('.zip')]
    
    scenarios=[]
    for z in zipfiles:
        zfile=zipfile.ZipFile(os.path.join(results_folder,z))
        statvarfile=[f.filename for f in zfile.infolist() if 'statvar' in f.filename][0]
        scenario=statvarfile.split(os.sep)[1].split('.')[1]
        if scenario not in scenarios:
            scenarios.append(scenario)
    return(scenarios)
    

def save_aggregated(var,aggregated_results_folder,GCMs,var_values,uzf_gage,**kwargs):
    # aggregated_results_folder: location to save aggregated files
    # GCMs: list of GCMs included in results. 
    # var_values: multi-level dict organized by [scenario][date][GCM]
    # uzf_gage: True or False
    
    try:
        separate=kwargs['separate']
    except KeyError:
        separate=False
        pass    
    
    # Function to write stuff out
    def write2output(outfile,GCMs,var_values,scenario):
        print ' -> saving to %s' %(outfile)
        ofp=open(outfile,'w')
        ofp.write('Date')
    
        # write headers
        for GCM in GCMs:
            ofp.write(',%s' %(GCM))
        ofp.write('\n')
    
        # write data, enforcing same column for each GCM
        if separate:
            timper=outfile.split('.')[-2]
            valuesdict=var_values[scenario][timper]
        else:
            valuesdict=var_values[scenario]
            
        for date in sorted(valuesdict.iterkeys()):
            ofp.write(str(date))
            for GCM in GCMs:
                try:
                    ofp.write(',%s' %(valuesdict[date][GCM][0]))
                except IndexError:
                    ofp.write(',NaN')
            ofp.write('\n')
        ofp.close()    

    #Fix code so that save works with separated csvs.
    if uzf_gage:
        var=var+'-uzfgage'
        
    #var_values[scenario][timeper][date][GCM]
    for scenario in var_values.iterkeys():
        if separate:
            for timeper in var_values[scenario].iterkeys():
                outfile=os.path.join(aggregated_results_folder,'%s.%s.%s.csv' %(var,scenario,timeper))
                
                write2output(outfile,GCMs,var_values,scenario)
        else:
            outfile=os.path.join(aggregated_results_folder,'%s.%s.csv' %(var,scenario)) 
        
            write2output(outfile,GCMs,var_values,scenario)


def aggregate(var,varlist,results_folder,mode,spinup,**kwargs):
    # var: variable to aggregate (can be a statvar, in the csv, or a gage/well)
    # varlist: list of all vars being aggregated for specified mode
    # results_folder: master folder of zipped results
    # mode: 'statvar','csvs','ggo','ssf'
    # separate=True/False # optional kwarg- if True, does not combine time periods when aggregating (for each scenario, sep. files for each time period)

    try:
        separate=kwargs['separate']
        print '(separate csv for each time period)'
    except KeyError:
        separate=False
        print 'time periods combined (one csv per %s variable-scenario)' %(mode)
        pass
    
    print var
    allfiles=os.listdir(results_folder)
    zipfiles=[f for f in allfiles if f.endswith('.zip')]    
    
    var_values=defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    if separate:
        var_values=defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
    overall_max=0
    overall_min=0
    GCMs=[]
    uzf_gage=False
    for z in zipfiles:
        
        zfile=zipfile.ZipFile(os.path.join(results_folder,z))
        statvarfile=[f for f in zfile.namelist() if 'statvar' in f][0]
        
        var_inds=np.where(np.array(varlist)==var)[0][0]
        # map results to GCM,scenario,realization,timeper, using statvar name
        GCM,scenario,realization,timeper=os.path.split(statvarfile)[-1].split('.')[:4]
        if GCM not in GCMs:
            GCMs.append(GCM)

        # set variable file and read parameters
        delim=None # default whitespace setting
        if mode=='statvar':
            varfile=statvarfile
            nvars=len(varlist)
            startrow=nvars+1
            var_inds=var_inds+7
        elif mode=='csv':
            varfile=[f for f in zfile.namelist() if '.csv' in f][0]
            startrow=1
            delim=','
            var_inds=var_inds+1
        elif mode=='ggo':
            # need to include language for identifying different types of ggo locations
            # e.g. UZF gages such as GaFF12-13
            varfile=var+'.ggo'
            startrow=2
            var_inds=2
        elif mode=='ssf':
            varfile=[f for f in zfile.namelist() if 'ssf' in f][0]
            startrow=0
            var_inds=3
        elif mode.lower()=='uzf':
            varfile=[f for f in zfile.namelist() if 'uzf' in f.lower()][0]
            startrow=3
            var_inds=var_inds+1
            
        # read in data from variable file
        varfile=zfile.open(varfile).readlines()
        t0=dt.datetime(int(timeper.split('-')[0]),1,1)
        ts=dt.datetime(int(timeper.split('-')[0])+spinup,1,1,0,0)
        columns=0
        for line in varfile[startrow:]:
            splitline=line.strip().split(delim)
            
            # record number of columns
            # if columns are missing at some point, file probably incomplete
            # (due to model run failure); stop reading
            if len(splitline)>columns:
                columns=len(splitline)
            if len(splitline)<columns:
                break
            
            if mode=='statvar':
                # working in datetime ensures proper sorting
                date=dt.datetime.strptime(' '.join(splitline[1:4]),'%Y %m %d') 
            elif mode=='csv':
                date=dt.datetime.strptime(splitline[0],'%m/%d/%Y')
            elif mode=='ggo':
                if len(splitline)==0:
                    uzf_gage=True # UZF gages have empty 3rd line
                    continue
                if uzf_gage:
                    date=t0+dt.timedelta(float(splitline[1])-1)
                else:
                    date=t0+dt.timedelta(float(splitline[0])-1)
            elif mode=='ssf':
                varline=splitline[0]
                if varline<>var:
                    continue
                date=dt.datetime.strptime(splitline[1],'%m/%d/%Y')-dt.timedelta(1)
            elif mode=='uzf':
                date=t0+dt.timedelta(float(splitline[0])-1)

            value=float(splitline[var_inds])
            
            if separate:
                valuesdict=var_values[scenario][timeper][date][GCM]
            else:
                valuesdict=var_values[scenario][date][GCM]
            
            if date>=ts: # if spinup period has ended, record
                valuesdict.append(value)
            else:
                continue
            # if overall max/min encountered, record
            if value> overall_max:
                overall_max=value
            if value< overall_min:
                overall_min=value
    return(overall_max,overall_min,GCMs,var_values,uzf_gage)


def get_var_files(var, aggregated_results_folder, **kwargs):
    
    try:
        separate = kwargs['separate']
    except KeyError:
        separate = False
        pass
    try:
        Scenarios=kwargs['Scenarios']
        if Scenarios==['all']:
            Scenarios='all'
    except KeyError:
        Scenarios='all'

    varfiles=[f for f in os.listdir(aggregated_results_folder) if var==f.strip().split('.')[0]]

    if not separate:
        num_scenarios=len(set([f.split('.')[-2] for f in varfiles]))
        varfiles=sorted(varfiles)[:num_scenarios] # in case var name is also contained in other var names, sorting should leave the simplest case at the top
    
    Files=[]
    if Scenarios<>'all':
        for scen in Scenarios:
            files=[f for f in varfiles if scen in f or '20c3m' in f]
            Files=Files+files
        varfiles=Files
        
    var_files_paths=[]
    for f in varfiles:
        var_files_paths.append(os.path.join(aggregated_results_folder,f))
    return var_files_paths

def modeselektor(mode):
    if mode=='ggo':
        stats=['mean_monthly', 'mean_annual', 'Q10', 'Q90']
    elif mode=='statvar' or mode=='csv':
        stats=['mean_monthly', 'mean_annual']
    elif mode=='uzf' or mode=='ssf':
        stats=['mean_annual']
    return(stats)

def plot_moving_avg_minmax(csvs,cols,timeunits,window,function,title,ylabel,colors,spinup,Synthetic_timepers):
    
    # csvs= list of csv files with multi-column timeseries
    # cols= list of column names to include in plot
    # timeunits= Pandas time units (e.g. 'D' for days)
    # window= width of moving avg window in timeunits
    # function= moving avg. fn to use (see Pandas doc)
    # title= plot title, ylabel= y-axis label
    # spinup= length of time (years) to trim off start of results when model is 'spinning up'

    # initialize plot
    fig=plt.figure()
    hatches=["","|","-",""]
    transp=[.3,.3,.3,.3]

    for i in range(len(csvs)):
        # load csvs into Pandas dataframes; reduce to columns of interest
        df=pd.read_csv(csvs[i], index_col='Date', parse_dates=True)
        try:
            df=df[cols]
        except KeyError:
            pass
        # trim spinup time from data to plot
        t0=df.index[0]
        tspinup=np.datetime64(t0+pd.DateOffset(years=spinup))        
        df=df[tspinup:]
        
        # resample at daily interval so that time gaps are filled with NaNs
        df_rs=df.resample('D') 
        
        # smooth each column with moving average
        smoothed=pd.rolling_window(df_rs,window,function,center='true')
           
        # plot out mean, max and min
        scenario = os.path.split(csvs[i])[1].split('.')[1]
        
        try:
            ax=smoothed.mean(axis=1).plot(color=colors[i],label=scenario)
        except TypeError:
            print "Problem plotting timeseries. Check that spinup value was not entered for plotting after spinup results already discarded during aggregation."
        
        ax.fill_between(smoothed.index,smoothed.max(axis=1),smoothed.min(axis=1),alpha=transp[i],color=colors[i],edgecolor='k',linewidth=0.25)
    
    # more plot settings
    wrap=60
    title="\n".join(textwrap.wrap(title, wrap)) #wrap title
    plt.subplots_adjust(top=0.85)        
    ax.set_title(title)
    
    ax.grid(False)
    handles,labels=ax.get_legend_handles_labels()
    if len(Synthetic_timepers)>0:
        handles.append(plt.Rectangle((0,0),1,1,color='0.9', linewidth=2))
        labels.append('synthetic input data')    
    ax.legend(handles,labels,title='Emissions scenarios',loc='best')
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
    window_yr=window/365.0
    ax.set_xlabel('Center of %s year moving window' %(window_yr))

    # shade periods for which synthetic data were generated
    if len(Synthetic_timepers)>0:
        for per in Synthetic_timepers:
            tstart,tend=map(int,per.split('-'))
            daterange=pd.date_range(start=dt.datetime(tstart,1,1),end=dt.datetime(tend,1,1))
            # make vectors of ymax values and ymin values for length of daterange
            ymax,ymin=np.ones(len(daterange))*plt.ylim()[1],np.ones(len(daterange))*plt.ylim()[0]
            syn=ax.fill_between(daterange,ymax,ymin,color='0.9',zorder=0)

def plot_q_minmax(csvs,cols,stat,title,ylabel,colors,spinup,scenarios,Synthetic_timepers):
    
    # csvs= list of csv files with multi-column timeseries
    # cols= list of column names to include in plot
    # stat= 'Mean Monthly', 'Mean Annual', 'Q10', 'Q90' 
    # title= plot title, ylabel= y-axis label
    # spinup= length of time (years) to trim off start of results when model is 'spinning up'
    
    # initialize plot
    fig=plt.figure()
    hatches=["","|","-",""]
    transp=[.3,.3,.3,.3]
        
    for i in range(len(csvs)):
        # load csvs into Pandas dataframes; reduce to columns of interest
        df=pd.read_csv(csvs[i], index_col='Date', parse_dates=True)
        try:
            df=df[cols]
        except KeyError: # in case not all of the scenarios are present
            pass
        
        # trim spinup time from data to plot
        t0=df.index[0]
        tspinup=np.datetime64(t0+pd.DateOffset(years=spinup))        
        df=df[tspinup:]
        
        # replace any zeros with NaNs
        df=df.replace({0:np.nan})
        
        # find any years with only one value (due to date convention)
        # reset values for those years to NaN
        an=df.groupby(lambda x: x.year)
        singles=[group for group in an.groups.keys() if len(an.get_group(group))==1]
        
        # calculate annual flow quantile
        if stat=='mean_annual':
            an=df.groupby(lambda x: x.year).mean()     
        elif stat=='Q90':
            an=df.groupby(lambda x: x.year).quantile(q=0.1,axis=0)
        elif stat=='Q10':
            an=df.groupby(lambda x: x.year).quantile(q=0.9,axis=0)
        
        # reset index from int to datetime (have to map to str first)
        an.index=pd.to_datetime(map(str,an.index)).values
        an=an.resample('AS')
        
        # blast years with only 1 daily value from above
        for s in singles:
            dstart,dend='%s-01-01'%(s),'%s-12-31' %(s)
            an[dstart:dend]=None
            
        # plot out mean, max and min
        scenario=[sc for sc in scenarios if sc in csvs[i]][0]
        ax=an.mean(axis=1).plot(color=colors[i],label=scenario)
        ax.fill_between(an.index,an.max(axis=1),an.min(axis=1),alpha=transp[i],color=colors[i],edgecolor='k',linewidth=0.25)   
            
    # more plot settings
    wrap=60
    title="\n".join(textwrap.wrap(title, wrap)) #wrap title
    plt.subplots_adjust(top=0.85)        
    ax.set_title(title)
    
    ax.grid(False)
    handles,labels=ax.get_legend_handles_labels()
    if len(Synthetic_timepers)>0:
        handles.append(plt.Rectangle((0,0),1,1,color='0.9', linewidth=2))
        labels.append('synthetic input data')
    ax.legend(handles,labels,title='Emissions scenarios',loc='best')
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
    #ax.set_xlabel('%s flow' %(stat))
    
    # shade periods for which synthetic data were generated
    if len(Synthetic_timepers)>0:
        for per in Synthetic_timepers:
            tstart,tend=map(int,per.split('-'))
            daterange=pd.date_range(start=dt.datetime(tstart,1,1),end=dt.datetime(tend,1,1))
            ymax,ymin=np.ones(len(daterange))*plt.ylim()[1],np.ones(len(daterange))*plt.ylim()[0]
            syn=ax.fill_between(daterange,ymax,ymin,color='0.9',zorder=0)     

def calc_boxstats(csvs, dates, ranges, baseline_dates, stat):
    # takes csv files for each variable (multiple scenarios) and aggregates
    # groups by month or by year and calculates period statistics
    # Average values for 20th century baseline are compared to future scenarios
    # Returns list columns concatenated by month (n dates to compare x 12 months; or n dates of annual averages to compare)
    # Each column consists of values produced by each GCM-scenario combination
    
    # Inputs:
    # csvs: list of csv files, one for each scenario
    # dates= list of dates compare (box for each date)
    # ranges= number of years to include in each box, centered around date
    # e.g. date: 2030, range: 10 = 2025-2035
    # baseline_dates: list with min,max for range of years to compare (e.g. [2055,2065])
    # stat= 'Mean Monthly', 'Mean Annual', 'Q10', 'Q90' 
    
    # initialize dict to store monthly groups, for each time period 
    groupsbyperiod=defaultdict(lambda: defaultdict(list))
    
    for d in range(len(dates)):
        
        # set time period limits
        tstart=np.datetime64('%s-01-01' %(int(dates[d]-0.5*ranges)))
        tstop=np.datetime64('%s-01-01' %(int(dates[d]+0.5*ranges)))-1 # get rid of last date, which by itself in a different year
        
        df=defaultdict()
        # build dict of Pandas dataframes, one for each csv.
        for i in range(len(csvs)):
            df[i]=pd.read_csv(csvs[i], index_col='Date', parse_dates=True)
            
            # check that tstart isn't within the specified model spinup period
            '''t0=df[i].index[0]
            tspinup=np.datetime64(t0+pd.DateOffset(years=spinup))
            if tstart<tspinup:
                tstart=tspinup
                print "Warning! Time period overlaps model spinup period, adjusting period start to end of spinup..."'''
            
            # Rename columns so they're unique.
            scenario=csvs[i].split('.')[-2]
            new_cols=[]
            for col_name in list(df[i].columns):
                new_cols.append('%s_%s' %(scenario,col_name))
            df[i].columns=new_cols
        
        # concatenate,resample, trim to tstart,tend
        cdf=pd.concat(df.values(),axis=1)
        cdf=cdf.resample('D')
        cdf=cdf[tstart:tstop]
        
        
        # group by month, add to dict
        if stat=='daily_means_by_month': # returns days x 1 array, for each month
            cdf_dm=cdf.mean(axis=1)
            cdf_dm_m=cdf_dm.groupby(lambda x: x.month)
            for month, group in cdf_dm_m:
                groupsbyperiod[d][month]=group.dropna() # dropna prob not needed due to prior avg
                
        elif stat=='mean_monthly': # returns 12 (months) x n GCM-scenarios array
            cdf_m=cdf.groupby(lambda x: x.month).mean()
            for month in cdf_m.transpose():
                groupsbyperiod[d][month]=cdf_m.transpose()[month].dropna()
                
        elif stat=='monthly_statistics': # returns days x n GCM-scenarios array, for each month
            cdf_m=cdf.groupby(lambda x: x.month)
            for month, group in cdf_m:
                reshaped=group.values.flatten() # flatten to 1-D columns
                dropnas=reshaped[~np.isnan(reshaped)] # drop NaNs
                groupsbyperiod[d][month]=dropnas[dropnas !=0] # drop 0s
                
        elif stat=='mean_annual':
            cdf_a=cdf.groupby(lambda x: x.year).mean()
            groupsbyperiod[d]=cdf_a.transpose().dropna().mean(axis=1)
        
        # population of quantiles from all scenario-gcm combinations; excludes 0 values        
        elif stat=='Q90':
            qts=cdf.replace({0:np.nan}).quantile(q=0.1,axis=0)
            groupsbyperiod[d]=qts.dropna()
        elif stat=='Q10':
            qts=cdf.replace({0:np.nan}).quantile(q=0.9,axis=0)
            groupsbyperiod[d]=qts.dropna()
               
    # collate time periods by stat time period
    collated=[]
    try: # for monthly plots
        for month in groupsbyperiod[d].iterkeys():
            for d in groupsbyperiod.iterkeys():
                collated.append(groupsbyperiod[d][month])
    except AttributeError: # annual plot (no months)
        for d in groupsbyperiod.iterkeys():
            collated.append(groupsbyperiod[d])
        
    # get 20th cent baseline data
    # should be mean monthly, or mean annual
    bstart=np.datetime64('%s-01-01' %(int(baseline_dates[0])))
    bstop=np.datetime64('%s-01-01' %(int(baseline_dates[1])))
    try:
        csv=[f for f in csvs if '20c3m' in f][0]
    except IndexError: # no file labeled '20c3m'- 20th cent probably included in continuous run
        # get 20th century data from one of the scenarios
        csv=csvs[0]

    df=pd.read_csv(csv, index_col='Date', parse_dates=True)
    dfm=df.mean(axis=1)[bstart:bstop] # mean across all GCMS for each date

    if stat=='mean_annual':
        dfm_annual=dfm.groupby(lambda x: x.year).mean()
        Baseline=[dfm_annual.mean()] # mean of annual means
    elif stat=='Q10':
        dfm_annual=dfm.groupby(lambda x: x.year).quantile(q=0.9,axis=0)
        Baseline=[dfm_annual.mean()] # mean of annual Q10 flows
        #Baseline=[dfm.replace({0:np.nan}).quantile(q=0.9,axis=0).median()]
    elif stat=='Q90':
        dfm_annual=dfm.groupby(lambda x: x.year).quantile(q=0.1,axis=0)
        Baseline=[dfm_annual.mean()] # mean of annual Q90 flows
        #Baseline=[dfm.replace({0:np.nan}).quantile(q=0.1,axis=0).median()]
    else:
        dfm_monthly=dfm.groupby(lambda x: x.month)
        Baseline=np.array(dfm_monthly.aggregate(np.mean)) # mean for each month
        
    return(collated,Baseline)


def box_plot(collated,Baseline,dates,ranges,baseline_dates,stat,title,ylabel):

    # collated= list of column vectors, collated by period (see dates and ranges below)
    # Baseline= array or list baseline averages by period (one for annual; 12 for monthly)
    # dates= list of dates to plot boxes for
    # ranges= number of years to include in each box, centered around date
    # e.g. date: 2030, range: 10 = 2025-2035
    # stat= 'Mean Monthly', 'Mean Annual', 'Q10', 'Q90'

    # initialize plot
    fig=plt.figure()
    ax=fig.add_subplot(111)
    
    # set box widths and positions
    if 'month' in stat:
        spacing = 0.1 # space between months
        boxwidth = (1-2*spacing)/len(dates)
        positions=[]
        for m in range(12):
            for d in range(len(dates)):
                position=0.5+m+spacing+(boxwidth*(d+0.5))
                positions.append(position)
    
    # make a box plot
    # mpl default for boxes extends to quartiles with median notch
    # according to mpl manual, "whis : [ default 1.5 ] Defines the length of the whiskers as a function of the inner quartile range. They extend to the most extreme data point within ( whis*(75%-25%) ) data range.
    # chose a value of 1.6 based on 90%-10% range (or 50%*1.6=80%)
    # not sure if this math is correct
    if 'month' in stat:
        bp=plt.boxplot(collated,positions=positions,widths=boxwidth,whis=1.6)
    else:
        bp=plt.boxplot(collated,whis=1.6)
        
    # make it look nice
    plt.setp(bp['medians'], color='black',linewidth=2)
    wrap=60
    title="\n".join(textwrap.wrap(title, wrap)) #wrap title
    fig.subplots_adjust(top=0.85)
    fig.subplots_adjust(bottom=0.25)
    
    ax.set_title(title)    
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(-3,3))
    #ax.set_xlabel('Month')
    
    # reset the xticks to one per month
    # should work for any number of date comparisons (e.g. 2060,2090...n)
    if 'month' in stat:
        frequency=1 # 1 for every month, 2 for every other, etc.
        ticks=(np.arange(12)+1)[frequency-1::frequency]
        ax.set_xticks(ticks)  
        months=[]
        for tick in ax.get_xticks():
            month=calendar.month_abbr[tick]
            months.append(month)
        ax.set_xticklabels(months)
    else: # for annual plots, set ticks based on year ranges for each box
        xlabels=[]
        for d in range(len(dates)):
            xl='%s-%s' %(int(dates[d]-0.5*ranges),int(dates[d]+0.5*ranges))
            xlabels.append(xl)
        ax.set_xticklabels(xlabels)
    
    # fill in the boxes
    boxColors = ['darkkhaki','royalblue','LawnGreen','DarkRed','Gold'] # html color names
    numBoxes = len(collated)
    medians = range(numBoxes)
    for i in range(numBoxes):
        box = bp['boxes'][i]
        boxX = []
        boxY = []
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
            boxCoords = zip(boxX,boxY)
        # Alternate between Dark Khaki and Royal Blue
        k = i % len(dates)
        boxPolygon = Polygon(boxCoords, facecolor=boxColors[k])
        ax.add_patch(boxPolygon)
        
        # add in line representing Baseline averages
        xmin,xmax=ax.get_xlim()
        xlength=xmax-xmin
        ax.axhline(Baseline[i/len(dates)], xmin=np.min(boxX)/xlength-xmin/xlength, xmax=np.max(boxX)/xlength-xmin/xlength, color='r', linewidth=2) # normalize to plot units
        
    # legend
    handels=[]
    labels=[]
    for d in range(len(dates)):
        handels.append(plt.Rectangle((0, 0), 1, 1, fc=boxColors[d]))
        labels.append('%s-%s' %(int(dates[d]-0.5*ranges),int(dates[d]+0.5*ranges)))
    handels.append(plt.Line2D(range(5), range(5), color='k', linewidth=2))
    labels.append('median')
    handels.append(plt.Line2D(range(10), range(10), color='r', linewidth=2))
    labels.append('mean %s for %s-%s (Baseline)' %(ylabel.split(',')[0],baseline_dates[0],baseline_dates[1]))
    handels.append(plt.Line2D(range(5), range(5), color='k', linewidth=0.5))
    labels.append('90th/10th percentiles; boxes indicate 75th and 25th percentiles')
    handels.append(plt.Line2D(range(2),range(2),linestyle='none',marker='+'))
    labels.append('outliers')
    ax.legend(handels,labels,fontsize=8,bbox_to_anchor=(0., -0.3, 1., .102), loc=4,ncol=2,borderaxespad=0.)


def set_plot_titles(var, mode, stat, var_info, aggregated_results_folder, plottype=None, quantile=None):
    # mode= type of output being processed: 'stavar' 'csv' 'ggo' 'ssf' or 'uzf'
    # stat= 'Mean Monthly', 'Mean Annual', 'Q10', 'Q90' 
    # make statvars dictionary with descriptions and units

    # Error in case no quantile value is given
    if stat == 'quantile' and not quantile:
        raise ValueError("stat = 'quantile' require that a float argument be entered for quantile, "
                         "e.g. quantile=0.1 for Q90 flow")

    def description(var, stat, plottype, var_info, quantile=None):

        '''
        Note: this algorithm still has a lot of holes, for BEC I had to do a lot of manual fixing of the variables table
        especially for storage variables, and variables related to snow
        also, the 'kludge' variables below weren't included in the automatic generation of the variables table,
        because their filenames were different than the PRMS variables file

        11/21/2014: this whole method needs to be re-written (and simplified!). Too much duct tape.
        '''

        if plottype.lower() == 'timeseries':

            # gages and head observations
            if var == 'head':
                ydescrip = 'Annual average water level'
                calc = 'mean' # pandas/numpy operation (e.g. 'mean' as in df.groupby(...).agg('mean')

            elif var == 'baseflow' or 'cfs' in var:

                if 'quantile' in stat and quantile==0.9:
                    ydescrip = 'High streamflow (Q$_{:.0f}$)'.format(100 * (1-quantile))
                    calc = 'quantile'
                elif 'quantile' in stat and quantile==0.1:
                    ydescrip = 'Low streamflow (Q$_{:.0f}$)'.format(100 * (1-quantile))
                    calc = 'quantile'
                else:
                    ydescrip = 'Annual flow'
                    calc = 'mean'

            # variables that should be summed for each time period
            # (e.g. converted from inches/day reported by model to in/month or in/year)
            elif 'annual' in stat and not var_info[var]['Desc'].split()[1].strip() == 'Storage':

                if var_info[var]['Units'] == 'inches':
                    ydescrip = 'Annual total'
                    calc = 'sum'

                else:
                    ydescrip = 'Annual average'
                    calc = 'mean'

            # otherwise, variable could be a flow rate or volume (in storage) that should be averaged
            else:
                ydescrip = 'Annual average'
                calc = 'mean'

        elif plottype.lower() == 'box' or plottype.lower() == 'violin':

            # variables that should be summed for each time period
            # (e.g. converted from inches/day reported by model to in/month or in/year)

            # gages and head observations
            if var == 'head':
                ydescrip = 'Average water level'
                calc = 'mean'

            elif 'quantile' in stat:

                if quantile==0.9:
                    ydescrip = 'High streamflow (Q$\mathregular{_{10}}$)'
                    calc = 'quantile'
                elif quantile==0.1:
                    ydescrip = 'Low streamflow (Q$\mathregular{_{90}}$)'
                    calc = 'quantile'
                else:
                    # this line doesn't work, need to figure out how to use TeX with variables
                    ydescrip = 'Q$_{{:.0f}}_0$)'.format(10 * (1-quantile))
                    calc = 'quantile'


            elif 'annual' in stat:

                # variables that should be summed for each time period
                # (e.g. converted from inches/day reported by model to in/month or in/year)
                if var == 'baseflow' or 'cfs' in var:
                    ydescrip = 'Annual streamflow'
                    calc = 'mean'

                elif var_info[var]['Units'] == 'inches' and not var_info[var]['Desc'].split()[1].strip() == 'Storage':
                    ydescrip = 'Average annual total'
                    calc = 'sum'
                    '''
                elif var == 'baseflow' or 'cfs' in var or 'flow rate' in var_info[var]['Desc']:
                    '''
                else:
                    '''
                    if 'quantile' in stat and quantile==0.9:
                        ydescrip = 'High streamflow (Q$_{:.0f}$)'.format(100 * (1-quantile))
                        calc = 'quantile'

                    elif 'quantile' in stat and quantile==0.1:
                        ydescrip = 'Low streamflow (Q$_{:.0f}$)'.format(100 * (1-quantile))
                        calc = 'quantile'
                    '''
                    if 'flow rate' in var_info[var]['Desc']:
                        ydescrip = 'Annual streamflow'
                        calc = 'mean'

                    else:
                        ydescrip = 'Annual average'
                        calc = 'mean'


            elif 'monthly' in stat:

                # variables that should be summed for each time period
                # (e.g. converted from inches/day reported by model to in/month or in/year)
                if var == 'baseflow' or 'cfs' in var:
                    ydescrip = 'Average monthly streamflow'
                    calc = 'mean'

                elif var_info[var]['Units'] == 'inches' and not var_info[var]['Desc'].split()[1].strip() == 'Storage':
                    ydescrip = 'Average monthly total'
                    calc = 'sum'

                elif 'flow rate' in var_info[var]['Desc']:
                    ydescrip = 'Average monthly streamflow'
                    calc = 'mean'

                else:
                    ydescrip = 'Monthly average'
                    calc = 'mean'

        return ydescrip, calc

    # file name is only needed to determine if gage package output represents streamgage or 'uzf' gage
    # if file for variable isn't in folder, pass over it
    try:
        fname = [f for f in os.listdir(aggregated_results_folder) if var == f.split('.')[0]][0]
    except:
        return

    # kludges to resolve differences between GSFLOW output variable names and the "Parameter file for PRMS/GSFLOW"
    # guessing with the basin_hortonian- the units (inches) in the var_name file make it seem like a PRMS variable,
    # but it is coming from the GSFLOW csv
    kludge = {'sat_stor': 'sat_store',
              'basinhortonian': 'basin_hortonian',
              'unsat_stor': 'unsat_store'}

    if var in kludge.keys():
        var = kludge[var]


    if mode =='csv' or mode == 'statvar':

        try:
            var_info[var]
            title = var_info[var]['Desc']
            units = var_info[var]['Units']
        except:
            if 'cfs' in var:
                title = var
                units = 'cubic feet per second'
            else:
                title, units = var, ""

        units = units\
            .replace('l3', 'cubic feet')\
            .replace('/t', ' per day')\
            .replace('temp_units', 'Fahrenheit')

        ydescrip, calc = description(var, stat, plottype, var_info, quantile)

    elif mode == 'ggo':
        if 'uzfgage' in fname:
            title = var
            units = 'feet'
            var = 'head'

        else:
            title = var
            units = 'cubic feet per day'
            var = 'baseflow'

        ydescrip, calc = description(var, stat, plottype, var_info, quantile)

    elif mode == 'ssf':
        title = var
        ydescrip = 'head'
        units = 'feet'
        var = 'head'
        ydescrip, calc = description(var, stat, plottype, var_info, quantile)

    elif mode == 'uzf':
        title = var
        ydescrip = var
        units = 'cubic feet per day'
        calc = 'sum'

    if len(ydescrip) == 0:
        ylabel = str(units).capitalize()
    else:
        ylabel ='%s, %s' %(ydescrip, units)
    xlabel = ''

    return title, xlabel, ylabel, calc


def get_var_info(var_name_file):

    varinfo = {}
    var_name_data = open(var_name_file, 'r').readlines()

    read = False
    for line in var_name_data:

        if 'name:' in line.lower():
            n = line.strip().split(':')[1].strip()
            read = True
            continue
        elif read and 'desc:' in line.lower():
            d = line.strip().split(':')[1].strip()

        elif read and 'units:' in line.lower():
            u = line.strip().split(':')[1].strip()
            varinfo[n.lower()] = {'Desc': d.lower(),
                            'Units': u.lower()}
            read = False

    return varinfo

def make_var_table(output_folder, var_name_file):

    var_info = get_var_info(var_name_file)

    ofp = open(os.path.join(output_folder, 'GSFLOW_variables.csv'), 'w')
    ofp.write('Output_file,variable,description,units,stat,calc,plot_type,ylabel_0,ylabel_1,xlabel,title\n')
    for mode in ['statvar', 'csv']:

        aggregated_results_folder = os.path.join(output_folder, mode)

        for var in var_info.iterkeys():

            for stat in ['mean_monthly', 'mean_annual']:
                if stat == 'mean_annual':
                    plots = ['timeseries', 'box']
                else:
                    plots = ['box']

                for plottype in plots:
                    try:
                        title, xlabel, ylabel, calc = set_plot_titles(var, mode, stat, var_info, aggregated_results_folder, plottype=plottype, quantile=None)
                    except TypeError: # means that no results file exists for the variable, skip it
                        continue
                    if not xlabel:
                        xlabel = ''

                    ofp.write('{0},{1},"{2}",{3},{4},{5},{6},{7},{8},"{9}"\n'.format(mode, var, var_info[var]['Desc'], var_info[var]['Units'], stat, calc, plottype, ylabel, xlabel, title))

    ofp.close()

