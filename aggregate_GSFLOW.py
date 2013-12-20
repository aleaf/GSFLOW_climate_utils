# aggregate and plot results from multiple GSFLOW climate scenario runs
# assumes that all results are in zip files (one per run) within a specified results_folder

import sys
import os
import numpy as np
from collections import defaultdict
import GSFLOW_utils

results_folder='results'
#mode='all' # 'stavar' 'csv' 'ggo' 'ssf' or 'uzf'
spinup=0 # number of years to trim from beginning of each dataset

try:
    mode=sys.argv[1] # 'statvar' 'csv' 'ggo' 'ssf' or 'uzf'
except IndexError:
    print '\naggregate+GSFLOW is called by entering:\n python aggregate+GSFLOW.py mode\n\nmode options: statvar, csv, ggo, ssf, uzf or all\n\nTo output each timeperiod to a separate csv, enter "separate" after the mode argument.\nOtherwise, time periods will be combined (one csv per variable-scenario).\n'
    quit()
 
if 'separate' in sys.argv:
    separate_flag=True
else:
    separate_flag=False


if mode=='all':
    print "Aggregating everything..."
    modes=['statvar', 'csv', 'ggo', 'ssf', 'uzf']
else:
    modes=[mode]

for m in modes:
    
    # check to see if output directory exists, if not, make one:
    if separate_flag:
        GSFLOW_utils.check4output_folder(m+'_separated')
        aggregated_results_folder=m+'_separated'
    else:
        GSFLOW_utils.check4output_folder(m)
        aggregated_results_folder=m
    
    print "getting list of variables..."
    varlist=GSFLOW_utils.getvars(results_folder,m)
    
    print '\nAggregating %s variables from all runs...' %(m)
    no_data=0
    for var in varlist:
        
        overall_max,overall_min,GCMs,var_values,uzf_gage=GSFLOW_utils.aggregate(var,varlist,results_folder,m,spinup,separate=separate_flag)
    
        # Test for all zeros (no data); if true skip to next var
        if overall_max==0 and overall_min==0:
            no_data+=1
            print " no data"
            continue
        
        GSFLOW_utils.save_aggregated(var,aggregated_results_folder,GCMs,var_values,uzf_gage,separate=separate_flag)
      
    print "%s %s had no data" %(no_data,m)
print "Done!"

    