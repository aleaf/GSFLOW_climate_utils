# Program to interpolate PRMS input data between discrete time periods
# interpolates daily means for temperatures (or other data) over an avg_per length (years)
# before and after a gap in data
# Discrete data such as precip can be copied to fill in the gap, favoring the data period after the gap
# (see hard-coded operations on lines 150-165)

import os
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict
import calendar
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pdb

# function to read dates from PRMS files
parse=lambda x: datetime.datetime.strptime(x,'%Y %m %d %H %M %S')


def setleapyearlims(yearlims):
    # function to set limits for interpolating the time gap (with leap years)
    # finds the next leap year after the start or before then end of yearlims
    leapyearlims=defaultdict()
    for i in range(2):
        if calendar.isleap(yearlims[i]):
            leapyearlims[i]=yearlims[i]
        else:
            inc=0
            for n in range(3):
                if i==0:
                    inc+=1 # move forwards if start year isn't a leap year
                else:
                    inc+=-1
                if calendar.isleap(yearlims[i]+inc):
                    leapyearlims[i]=yearlims[i]+inc
                else:
                    continue
    return(leapyearlims.values()) 


def read_datafile(datadir,f):
    print f
    timeper=f.split('.')[3]
    
    # find first line of data; record header information
    temp=open(os.path.join(datadir,f)).readlines()
    header=[]
    Ncolumns=defaultdict()
    for line in temp:
        try:
            int(line.split()[0])
            break
        except:
            try:
                ncols=int(line.strip().split()[1])
                var=line.split()[0]
                Ncolumns[var]=ncols
                header.append(line.strip())
                continue
            except:
                header.append(line.strip())
                continue         
    
    # read file into pandas dataframe (index_col is kludgy but works)
    df=pd.read_csv(os.path.join(datadir,f),delim_whitespace=True,dtype=None,header=None,skiprows=len(header),parse_dates=[[0,1,2,3,4,5]],date_parser=parse,index_col='0_1_2_3_4_5')
    
    return df,timeper,Ncolumns,header

def parseheader(header):
    for line in header:
        if 'tmin' in line:
            tmin=int(line.strip().split()[1])
        elif 'tmax' in line:
            tmax=int(line.strip().split()[1])
        elif 'precip' in line:
            precip=int(line.strip().split()[1])
    return tmin,tmax,precip


def read_PRMS(files,datadir,avg_per,std_rolling_mean_window):
    
    timepers=sorted([f.split('.')[3] for f in files])
    starts=defaultdict()
    ends=defaultdict()
    Std=defaultdict()
    Std_sm=defaultdict()
    data=defaultdict()
    
    for f in files:
        print f
        timeper=f.split('.')[3]
        
        df,timeper,Ncolumns,header=read_datafile(datadir,f)
        
        data[timeper]=df # save data for timeper to dict for later
        # calculate daily means before and after the time gap
        starts,ends=calc_daily_means(df,timeper,timepers,starts,ends,avg_per)
        # calculate daily standard deviations before and after the time gap
        Std[timeper]=df.groupby([lambda x: x.month,lambda x: x.day]).std()
        
        # smooth standard deviations using rolling mean (moving window)
        # start by extending daily std values by 1/2 of window size
        insert,append = 0.5*std_rolling_mean_window,0.5*std_rolling_mean_window-1
        Std_extended=Std[timeper][-insert:].append(Std[timeper]).append(Std[timeper][0:append])
        Std_smoothed=pd.rolling_window(Std_extended,std_rolling_mean_window,'boxcar',center='true')
        Std_sm[timeper]=Std_smoothed[insert:-append]
        
    return(data,timepers,starts,ends,Std,Std_sm,Ncolumns,header)


def calc_daily_means(df,timeper,timepers,starts,ends,avg_per):
    # calculate daily means for first and/or last 10 years

    # define the starts/ends of averaging periods
    endavg_start=np.datetime64(df.index[-1]-pd.DateOffset(years=avg_per))
    beginavg_end=np.datetime64(df.index[0]+pd.DateOffset(years=avg_per))
    
    if timeper in timepers[:-1]: # average the ends
        daily_means_end=df[endavg_start:df.index[-1]].groupby([lambda x: x.month,lambda x: x.day]).mean()
        ends[timeper]=daily_means_end
    if timeper in timepers[1:]: # average the starts
        daily_means_start=df[df.index[0]:beginavg_end].groupby([lambda x: x.month,lambda x: x.day]).mean()
        starts[timeper]=daily_means_start
    return starts,ends


def diff_from_daily_mean(dailymeans,dailyvalues,tstart,tend):
    # first build dataframe of copied daily means, same shape as dailyvalues
    allmeans=pd.DataFrame()
    for i in range(tend+1)[tstart:]:
        datez=pd.date_range(start=datetime.datetime(i,1,1),end=datetime.datetime(i,12,31))
        if calendar.isleap(i):
            means366=dailymeans.copy()
            means366.index=datez
            allmeans=allmeans.append(means366)
        else:
            means365=dailymeans[(1,1):(2,28)].append(dailymeans[(3,1):(21,31)])
            means365=means365.copy()
            means365.index=datez
            allmeans=allmeans.append(means365)
    start=pd.Timestamp(datetime.datetime(tstart,1,1))
    end=pd.Timestamp(datetime.datetime(tend,12,31))
    dailyvalues_trim=dailyvalues[start:end]
    diffs=dailyvalues_trim-allmeans
    
    return diffs


def get_before_after_periods(gap,timepers,i):

    gaplength=int(gap[-4:])-int(gap[0:4])-1
    splityear=int(round(int(gap[0:4])+0.5*gaplength,0))
    len1,len2=splityear-int(gap[0:4])-1,int(gap[-4:])-splityear

    #before
    tend1=int(timepers[i-1][-4:])
    tstart1=tend1-len1
    beforeperstart=int(timepers[i-1][0:4])
    if tstart1<beforeperstart:
        tstart1=beforeperstart
        len2=gaplength-(tend1-tstart1)
    #after
    tstart2=int(timepers[i][0:4])
    tend2=tstart2+len2
    afterperend=int(timepers[i][-4:])
    print 'tend2=%s afterperend= %s' %(tend2,afterperend)
    if tend2>afterperend:
        tend2=afterperend
        len2=tend2-tstart2+1
        print 'gaplength=%s, tend1=%s, len1=%s' %(gaplength,tend1,gaplength-len2)
        tstart1=tend1+1-(gaplength-len2)
        
    print '%s,%s,%s,%s' %(tstart1,tend1,tstart2,tend2)    
    return splityear,tstart1,tend1,tstart2,tend2
    
    
def apply_variability_from_ends(i,gap,timepers,t0,t1,data,df0,Temp_columns):
    # get diffs before and after gap
    print 'applying variability from ends...'
    splityear,tstart1,tend1,tstart2,tend2=get_before_after_periods(gap,timepers,i)
    
    diffs1=diff_from_daily_mean(t0,data[timepers[i-1]],tstart1,tend1)
    print diffs1.index
    # reindex
    newstart=diffs1.index[-1]+datetime.timedelta(1)
    newend=newstart+datetime.timedelta(len(diffs1.index)-1)
    diffs1.index=pd.date_range(start=newstart,end=newend)
    print diffs1.index
    

    diffs2=diff_from_daily_mean(t1,data[timepers[i]],tstart2,tend2)
    print diffs2.index
    #reindex
    newstart=newend+datetime.timedelta(1)
    newend=newstart+datetime.timedelta(len(diffs2.index)-1)
    diffs2.index=pd.date_range(start=newstart,end=newend)
    print diffs2.index
    
    diffs=diffs1.append(diffs2)
    print diffs.index
    diffs=diffs[diffs.columns[0:Temp_columns]].sort() # trim precip
    df0=df0+diffs # add the differences on to the interpolated daily means
    
    return df0


def interp_PRMS(data,timepers,avg_per,Ncolumns,starts,ends,Std,Noise,Int_method,Copy_var):
    Means=defaultdict()
    for i in range(len(timepers))[1:]:
        t1=starts[timepers[i]]
        t0=ends[timepers[i-1]]
        var1,var0=Std[timepers[i]],Std[timepers[i-1]] # carrying the daily variance values along with means
        endyear=int(timepers[i].split('-')[0])-1 # minus 1, otherwise will overlap
        startyear=int(timepers[i-1].split('-')[1])+1
        gap='%s-%s' %(timepers[i-1].split('-')[1],timepers[i].split('-')[0])
        print gap
    
        # set start/end for interpolation to midpoints of periods incorporated in daily averages
        yearlims=int(startyear-0.5*avg_per),int(endyear+0.5*avg_per)
        leapyearlims=setleapyearlims(yearlims)
        
        for month,day in t0.index:
            
            if (month,day)==(2,29):
                d0=np.datetime64(datetime.datetime(leapyearlims[0],month,day))
                d1=np.datetime64(datetime.datetime(leapyearlims[1],month,day))
            else:
                d0=np.datetime64(datetime.datetime(yearlims[0],month,day))
                d1=np.datetime64(datetime.datetime(yearlims[1],month,day))
                
            v0s=t0[(month,day):(month,day)]
            var0s=var0[(month,day):(month,day)]
            
            
            if Int_method=='DailyMeans':
                v1s=t1[(month,day):(month,day)]
                var1s=var1[(month,day):(month,day)]
            # Option2- interpolate based on change in mean annual temps between the two periods
            # use annual means for tmin,tmax at each station
            # (to try to smooth out exaggerated trends in individual days)
            elif Int_method=='AnnualMean':
                dAnnualMeans=t1.mean()-t0.mean() # annual means for each station
                # instead calculate daily values after the gap by adding dAnnualMeans to daily means before gap
                v1s=t0[(month,day):(month,day)]+dAnnualMeans
                # do the same for the standard dev.
                dMeanVar=var1.mean()-var0.mean()
                var1s=var0[(month,day):(month,day)]+dMeanVar
            
            # create new dataframe with daily values on either side of gap
            df=v0s.append(v1s)
            df.index=[d0,d1]
            dfv=var0s.append(var1s)
            dfv.index=[d0,d1]
            
            # resample to fill in gap with annual index intervals
            df=df.resample('AS')
            df.index=df.index+pd.DateOffset(month=month,day=day)
            dfv=dfv.resample('AS')
            dfv.index=df.index
            
            # linear interpolation over new index
            df=df.apply(pd.Series.interpolate)
            dfv=dfv.apply(pd.Series.interpolate)
       
            # combine interpolated days
            if month==1 and day==1:
                df0=df
                dfv0=dfv
            else:
                df0=df0.append(df)
                dfv0=dfv0.append(dfv)
                
        # Trim interpolated data to start/end of gap
        print 'df0 index= %s' %(df0.index)
        start,end=datetime.datetime(startyear,1,1),datetime.datetime(endyear,12,31)
        df0=df0.sort()[start:end]
        dfv0=dfv0.sort()[start:end]
        print 'df0 index= %s' %(df0.index)
        # Trim interpolated data to only temps; sort and resample to remove extra 2-28s
        Temp_columns=Ncolumns['tmax']+Ncolumns['tmin']    
        df0=df0[df0.columns[0:Temp_columns]].sort().resample('D')
        print 'df0 index= %s' %(df0.index)
        dfv0=dfv0[dfv0.columns[0:Temp_columns]].sort().resample('D') # trim standard deviations too

        # Option 1 for daily variability: calculate random noise for each day using the standard devs. for real data periods
        rows,cols=np.shape(df0)
        noise=np.random.randn(rows) # one random value for tmin,tmax for all stations (because they are not independant!)
        noise=pd.Series(noise,index=df0.index)
        # function to multily (columnwise) station daily standard deviations by noise
        func=lambda x: np.asarray(x) * np.asarray(noise)
        
        if Noise or Copy_var:
            dfm=df0 # make a copy of df0, so daily means can be compared to synthetic data with noise
        if Noise:
            df0=df0+dfv0.apply(func) # means + stdevs*random
        
        if Copy_var:
        # Option 2 for daily variability: get differences from the daily means before and after the gap
        # and apply these differences to the interpolated daily means to get the synthetic values
            df0=apply_variability_from_ends(i,gap,timepers,t0,t1,data,df0,Temp_columns)
            
        # Join precip back on to end of dataframe for gap;
        # Get precip by copying data from two adjacent periods, taking all of the second
        # period, and as much of the first period as needed to fill
        prcp_cols=data[timepers[i]].columns[Temp_columns:]

        splityear,tstart1,tend1,tstart2,tend2=get_before_after_periods(gap,timepers,i)        
        
        tstart1=datetime.datetime(tstart1,1,1)
        tend1=datetime.datetime(tend1,12,31)
        tstart2=datetime.datetime(tstart2,1,1)
        tend2=datetime.datetime(tend2,12,31)
        
        
        prcpStart=data[timepers[i-1]][prcp_cols][tstart1:tend1]
        prcpEnd=data[timepers[i]][prcp_cols][tstart2:tend2]
        dfP=prcpStart.append(prcpEnd).sort().copy()
        #dfP_trim=dfP[-len(df0):]
        dfP_trim=dfP[0:len(df0)]
        print dfP_trim.index
        dfP_trim.index=df0.index
        
        temp=pd.merge(df0,dfP_trim,left_index=True,right_index=True)
        data[gap]=temp.resample('D')
        
        # store daily means for gaps in seperate dictionary
        if Noise or Copy_var:
            Means[gap]=dfm
    
    
    print '\nreassembling...'
    df=data[sorted(data.keys())[0]]
    for per in sorted(data.keys())[1:]:
        print df.index
        print per
        df=df.append(data[per])
            
    if Noise or Copy_var:
        dfm=Means[sorted(Means.keys())[0]]
        for per in sorted(Means.keys())[1:]:
            dfm=dfm.append(Means[per])
        dfm=dfm.resample('D')
    else:
        dfm=False
        
    Newtimepers=sorted(data.keys())
    
    # sort and resample to daily intervals
    # (this gets rid of extra 2-28 values, which arise from interpolating 2-29s)
    # for years with duplicate 2-28s, df.resample simply takes the mean of the two values
    df=df.sort().resample('D')
    
    return df,Newtimepers,dfm
       
       
def make_plots(df,dfm,Std,Std_sm,std_rolling_mean_window,timepers,Ncolumns,outpdf):
    
    def plotdfm(dfm,inds,label):
        try:
            dfm[inds].plot(label=label,color='r')
        except:
            pass

    plt.figure()
    df[6].plot(label='Tmax - synthetic data')
    plotdfm(dfm,6,'Tmax - interpolated daily means')
    plt.legend()
    outpdf.savefig()
    plt.figure()
    df[6+19].plot(label='Tmin - synthetic data')
    plotdfm(dfm,6+Ncolumns['tmax'],'Tmin - interpolated daily means')
    plt.legend()
    outpdf.savefig()
    plt.figure()
    df[6+19*2].plot(label='Precip - synthetic data')
    plotdfm(dfm,6+Ncolumns['tmax']+Ncolumns['tmin'],'Tmax')
    plt.legend()
    outpdf.savefig()
    
    # make a PDF to look at the annual variation in stdev for the variables
    # compare actual stdevs to moving averages
    vars=['tmax','tmin']
    columns=[6,6+Ncolumns['tmax']]
    for timeper in timepers:
        plt.figure()
        for i in range(len(vars)):
            plt.plot(Std[timeper][columns[i]],label='%s computed standard deviation' %(vars[i]))
            plt.plot(Std_sm[timeper][columns[i]],label='%s %s day rolling mean of stdev' %(vars[i],std_rolling_mean_window))
            plt.title('Daily standard deviation in %s for %s' %(','.join(vars),timeper))
        plt.legend()
        outpdf.savefig()
    outpdf.close()

def writeoutput(df,header,outfile,newfilesdir,fmt):
    ofp=open(os.path.join(newfilesdir,outfile),'w')
    
    print '\nwriting output to %s\n' %(outfile)
    
    ofp.write('created by PRMS_data_interp.py\n')
    for lines in header[1:]:
        ofp.write(lines+'\n')
    
    for i in range(len(df)):
        ofp.write('%s %s %s 0 0 0  ' %(df.index[i].year,df.index[i].month,df.index[i].day))
        ofp.write(' '.join(map(fmt,list(df[i:i+1].values[0]))))
        ofp.write('\n')
    ofp.close()
