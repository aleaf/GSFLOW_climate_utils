# aggregate and plot results from statvar files for multiple GSFLOW climate scenario runs
# assumes that all results are in zip files (one per run) within a specified results folder

import sys
import os
import numpy as np
import pandas as pd
import GSFLOW_utils
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pdb

message='\nGSFLOW_plots is called by entering:\n' \
        'python GSFLOW_plots.py mode stat\n\n' \
        'mode options: statvar, csv, ggo, ssf or uzf\n' \
        'stat options: mean_monthly, mean_annual, Q10, Q90\n' \
        '-or-\n' \
        'To plot everything, simply type "all."\n\n' \
        'To plot time periods separately, enter "separate" after mode and stat.\n'

try:
    mode=sys.argv[1] # 'statvar' 'csv' 'ggo' 'ssf' or 'uzf'
except IndexError:
    print message
    quit()

if 'separate' in sys.argv:
    separate_flag=True
else:
    separate_flag=False

if mode=='all':
    All=True
    print "\nPlotting everything..."
    modes=['statvar','csv', 'ggo', 'ssf', 'uzf']
else:
    modes=[mode]
    All=False
    try: 
        stats=[sys.argv[2]] # stat= 'mean_monthly', 'mean_annual', 'Q10', 'Q90'
    except IndexError:
        print message
        quit()
     
#modes=['statvar']
#stats=['mean_annual']
#All=False
#separate_flag=False # in case you would like the timeperiods plotted individually

# input files
results_folder='results'
aggregated_results_folder_extension='' # appended to mode to get aggregated results folder e.g. 'statvar_run1'
var_name_file='BEC.var_name' # for statvar titles
plots_folder='plots_run3'

# check for input files
try:
    with open(var_name_file): pass
except IOError:
    print 'GSFLOW var_name file (descriptions of the variables) not found! Please get one and then try again.'

# GCMs to include in moving average envelope plots
GCMs2include=['cccma_cgcm3_1','cccma_cgcm3_1_t63','cnrm_cm3','csiro_mk3_0','csiro_mk3_5','gfdl_cm2_0','giss_aom','giss_model_e_r','iap_fgoals1_0_g','miroc3_2_hires','miub_echo_g','mpi_echam5','mri_cgcm2_3_2a']
Scenarios2include=['all'] # enter either 'all' or a list of specific scenarios

# moving average plot settings
timeunits='D' # pandas time units
window=365*3 # width of moving average window in tunits
function='boxcar' # moving average function to use (see Pandas doc)
spinup=0 # years to trim from start of model simulation, to allow for model spinup
Synthetic_timepers=['2001-2045','2066-2080'] # periods for which synthetic data were generated (to denote on plots)
minmaxcolors=["r", "b", "y",'0.5']

# Box plot settings
box_dates=[2062.5,2097.5] # list of integers representing years
ranges=5 # years to summarize in box plots, centered around box_dates
baseline_dates=[1995,2000] # [start,end] for years to include in baseline

for mode in modes:
    
    if All:
        stats=GSFLOW_utils.modeselektor(mode)
    
    for stat in stats:
        print 120*'*'
        print "\nMode: %s" %(mode)
        print "Statistics to plot:",
        for s in stats:
            print s,
        
        aggregated_results_folder=mode+aggregated_results_folder_extension # mode names are based on locations of aggregated results
        print '\npulling files from aggregated results folder: %s' %(aggregated_results_folder)
         
        # Output files
        moving_avg_plots=mode+'_moving_avg_plots.pdf'
        box_plots=mode+'_'+stat+'_box_plots.pdf'
        annual_min_max_plots=mode+'_'+stat+'_annual_min_max.pdf'
        
        print "\ngetting lists of variables and scenarios..."
        varlist=GSFLOW_utils.getvars(aggregated_results_folder,mode)
        #varlist=[var for var in varlist if var=='basin_ppt']
        scenarios=GSFLOW_utils.getscenarios(results_folder)
        
        print "\nPlotting aggregated %s results for..." %(stat)

        if separate_flag:
            print "(one plot per time variable-time period)"

            # make a plots folder if there isn't one already            
            GSFLOW_utils.check4output_folder(plots_folder+'_sep')

            # Check for aggregated results
            aggregated_results_folder=mode+'_separated'
            try:
                os.listdir(aggregated_results_folder)
            except:
                print '%s results have not been aggregated separately by time period! Cannot plot time periods separately. Please run aggregate_GSFLOW.py in "separate" mode.' %(mode)
                quit()

            # initialize plot
            annual_min_max=PdfPages(os.path.join(plots_folder+'_sep',annual_min_max_plots))

            for var in varlist:
                print var,

                # Get list of files for variable
                var_files_paths=GSFLOW_utils.get_var_files(var,aggregated_results_folder,separate=separate_flag,Scenarios=Scenarios2include)

                timpers=np.unique([f.split('.')[-2] for f in var_files_paths])

                print '\tMin/max plots:'
                for t in timpers:
                    print '\t%s' %(t)

                    # Set plot titles and ylabels
                    title,box_ylabel,mov_ylabel=GSFLOW_utils.set_plot_titles(var,mode,stat,varlist,var_name_file,aggregated_results_folder)

                    GSFLOW_utils.plot_q_minmax(csvs,GCMs2include,stat,title,box_ylabel,minmaxcolors,spinup,scenarios,Synthetic_timepers)
                    annual_min_max.savefig()

            annual_min_max.close()
            print 'Done!'
            quit()

        if not separate_flag:

            # make a plots folder if there isn't one already
            GSFLOW_utils.check4output_folder(plots_folder)  
            
            # initialize plots
            moving_avg=PdfPages(os.path.join(plots_folder,moving_avg_plots))
            box=PdfPages(os.path.join(plots_folder,box_plots))
            annual_min_max=PdfPages(os.path.join(plots_folder,annual_min_max_plots))
            
            for var in varlist:
                print var

                # Get list of files for variable
                var_files_paths=GSFLOW_utils.get_var_files(var,aggregated_results_folder,Scenarios=Scenarios2include)

                # Set plot titles and ylabels
                title,box_ylabel,mov_ylabel=GSFLOW_utils.set_plot_titles(var,mode,stat,varlist,var_name_file,aggregated_results_folder)

                print '\t*Moving average min/max plot',
                GSFLOW_utils.plot_moving_avg_minmax(var_files_paths,GCMs2include,timeunits,window,function,title,mov_ylabel,minmaxcolors,spinup,Synthetic_timepers)
                moving_avg.savefig()
                
                print '\t...Calculating %s for box plot...' %(box_ylabel.lower()),
                collated,Baseline=GSFLOW_utils.calc_boxstats(var_files_paths,box_dates,ranges,baseline_dates,stat)
                
                # if model run terminated early, there might not be any data for box plots
                box_data=True
                try:
                    if np.max([len(l) for l in collated])==0:
                        box_data=False
                except:
                    if len(collated)==0:
                        box_data=False
       
                if box_data:
                    print '\t*Box plot'
                    GSFLOW_utils.box_plot(collated,Baseline,box_dates,ranges,baseline_dates,stat,title,box_ylabel)
                    box.savefig()

                if 'Q' in stat or 'annual' in stat:
                    if 'Q' in stat and 'uzfgage' in var_files_paths:
                        continue
                    else:
                        print '\t*Min/max plot for Q90 and Q10 flows'
                        GSFLOW_utils.plot_q_minmax(var_files_paths,GCMs2include,stat,title,box_ylabel,minmaxcolors,spinup,scenarios,Synthetic_timepers)
                        annual_min_max.savefig()

            annual_min_max.close()  
            moving_avg.close()
            box.close()
        print "\nDone with %s %s plots" %(mode,stat)
    print "\nDone with all %s plots" %(mode)
print "\nFinished OK"
