# Program to interpolate PRMS input data between discrete time periods
# interpolates daily means for temperatures (or other data) over an avg_per length (years)
# before and after a gap in data
# Discrete data such as precip can be copied to fill in the gap, favoring the data period after the gap
# (see hard-coded operations on lines 150-165)
#
# Interpolation methods:
# AnnualMean: calculates change in temperature across the datagap by adding changes in annual mean temp (one for each station)
# to daily means before the gap
# DailyMean: calculates change in temp across the datagap using daily means after the gap. This method is problematic, in that
# each day of the year is allowed to have a different trend. This was causing the growing season to shrink even when annual mean temp was increasing.


import os
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import PRMS_data_interp_functions
import pdb

datadir='input_no_spinup' # dir with datafiles for discrete periods
newfilesdir='continuous_input2' # where to place new files with continuous input data
plotsdir=newfilesdir+'_plots' # plots of the new data

avg_per=20 # length of time (years) over which to calculate daily means for generation of annual data
Int_method='AnnualMean' # 'AnnualMean' or 'DailyMeans' # a more elegant way to do it would be to calculate a moving mean (to smooth out rouge days)

# use random noise to simulate daily variabiliy (Noise=False to not do this)
Noise=False # add random noise to interpolated daily means; magnitude of noise based on daily standard deviations in tmin,tmax
use_smoothed=False # use smoothed values for daily standard deviations in temperature
std_rolling_mean_window=100 # width of moving window (days) for smoothing daily standard deviations via rolling mean

# instead of random noise, simulate daily variability by copying temp deviations from daily means for periods with data (just like for precip)
Copy_var=True

datafiles=[f for f in os.listdir(datadir) if f.endswith('.data')]
GCM_scenarios=np.unique(['.'.join(f.split('.')[0:2]) for f in datafiles if '20c3m' not in f])

# MAIN program
print 'grouping data by GCM-scenario:\n'

for g in GCM_scenarios[0:1]:

    print '%s\n' %(g)
    # list files for each GCM-Scenario, including 20th cent
    GCM=g.split('.')[0]
    scenario=g.split('.')[1]
    files=[f for f in datafiles if g in f]
    _20thcent=[f for f in datafiles if GCM==f.split('.')[0] and '20c3m' in f]
    files=sorted(files+_20thcent)
    
    print 'reading in original PRMS data files...'
    data,timepers,starts,ends,Std,Std_sm,Ncolumns,header=PRMS_data_interp_functions.read_PRMS(files,datadir,avg_per,std_rolling_mean_window)
    
    print '\ninterpolating data across time gaps using %s from first and last %s years...' %(Int_method,avg_per)
    if use_smoothed:
        df,Newtimepers,dfm=PRMS_data_interp_functions.interp_PRMS(data,timepers,avg_per,Ncolumns,starts,ends,Std_sm,Noise,Int_method,Copy_var)
    else:
        df,Newtimepers,dfm=PRMS_data_interp_functions.interp_PRMS(data,timepers,avg_per,Ncolumns,starts,ends,Std,Noise,Int_method,Copy_var)
    
    # make some quick plots
    outpdf=PdfPages(os.path.join(plotsdir,'%s.pdf' %(g)))
    PRMS_data_interp_functions.make_plots(df,dfm,Std,Std_sm,std_rolling_mean_window,timepers,Ncolumns,outpdf)        
    print '\nsaving plots to %s.pdf\n' %(g)
    
    # write to PRMS data file
    fmt='{:.2f}'.format
    newtimeper='%s-%s' %(Newtimepers[0][0:4],Newtimepers[-1][-4:])
    outfile='%s.%s.1.%s.bec_ide.data' %(GCM,scenario,newtimeper)
    PRMS_data_interp_functions.writeoutput(df,header,outfile,newfilesdir,fmt)

print 'Done'
    
    
    
    