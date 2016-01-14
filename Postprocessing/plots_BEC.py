__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import climate_plots as cp

# input
results_path = '/Users/aleaf/Documents/BlackEarth/run3'
modes = ['ggo_diff', 'ggo_cfs', 'csv', 'statvar', 'ssf', 'uzf'] #['statvar', 'csv', 'ggo', 'ssf', 'uzf']
var_name_file = 'BEC.var_name' # descriptions of GSFLOW variables

# create a variables table by running GSFLOW_utils.make_var_table()
# this table can then be edited to customize plot titles and y labels
GSFLOW_variables_table = '/Users/aleaf/Documents/BlackEarth/run3/GSFLOW_variables.csv'


# output
output_folder = '/Users/aleaf/Documents/BlackEarth/run3/plots_run3_3'


# exclude any GSFLOW/PRMS variables in this list
exclude = ['basinnetgwwel', 'obs_strmflow', 'kkiter', 'basin_horad']

# GCMs and scenarios to include in moving average envelope plots
gcms = ['cccma_cgcm3_1',
              'cccma_cgcm3_1_t63',
              'cnrm_cm3',
              'csiro_mk3_0',
              'csiro_mk3_5',
              'gfdl_cm2_0',
              'giss_aom',
              'giss_model_e_r',
              'iap_fgoals1_0_g',
              'miroc3_2_hires',
              'miub_echo_g',
              'mpi_echam5',
              'mri_cgcm2_3_2a']

Scenarios2include = ['sresa1b', 'sresa2', 'sresb1']

# time series plot settings
spinup = 20 # years to trim from start of model simulation, to discard data from model 'spinup' period

# list of pandas date_ranges, for periods for which synthetic data were generated (to denote on plots);
synthetic_timepers = [pd.date_range('{}-01-01'.format(2001), '{}-12-31'.format(2045)),
                      pd.date_range('{}-01-01'.format(2065), '{}-12-31'.format(2080))]


# use negative values for zorders so that shaded "fill-betweens" don't cover up tick marks
timeseries_properties = {'sresa1b': {'color': 'Tomato', 'zorder': -2, 'alpha': 1, 'lw': 1.5}, # (146, 0, 0)
                         'sresa2': {'color': 'SteelBlue', 'zorder': -3, 'alpha': 1, 'lw': 1.5}, #(0, 109, 219)
                         'sresb1': {'color': 'Yellow', 'zorder': -1, 'alpha': 1, 'lw': 1.5},
                         }

# customize any default rc settings
plotstyle = {'xtick.minor.size': 0}


# Box plot settings
compare_periods = np.array([[2060, 2065], [2095, 2100]]) # array
baseline_period = np.array([1995, 2000]) # array ([start, end]) for years to include in baseline

vars = pd.read_csv(GSFLOW_variables_table)

for mode in modes:

    results_folder = os.path.join(results_path, mode)

    # Instantiate report figures class
    Figs = cp.ReportFigures(mode, compare_periods, baseline_period, gcms, spinup,
                            results_folder, output_folder,
                            var_name_file,
                            timeseries_properties,
                            variables_table=vars,
                            synthetic_timepers=synthetic_timepers,
                            exclude=exclude,
                            plotstyle=plotstyle)

    # Make individual legends for each kind of plot
    Figs.make_box_legend()
    Figs.make_violin_legend()
    #Figs.make_timeseries_legend(baseline=True)
    Figs.make_timeseries_hexbin_legend()

    # For each variable (or model output observation), make the plots
    for var in Figs.varlist:

        print '\n{}'.format(var)

        # Make {scenario: csv} dictionary of the csv files for variable or observation
        csvs = dict([(scen, os.path.join(results_folder, f))
                     for scen in Scenarios2include for f in os.listdir(results_folder)
                     if f.split('.')[0] == var and scen in f])

        # Monthly Flows
        #print 'Box plot of monthly flows'
        #Figs.make_box(csvs, var, 'mean_monthly')

        # Annual Flows
        for stat in ['mean_annual', 'quantile']:

            if 'Q' in stat and 'uzfgage' in csvs:
                continue

            elif stat == 'mean_annual':
                print 'Annual Flows: ',

                print 'Time series ',
                #Figs.make_timeseries(csvs, var, stat, baseline=True)
                Figs.make_timeseries_hexbin(csvs, var, stat, baseline=True)

                #print 'Violin ',
                #Figs.make_violin(csvs, var, stat)

                #print 'Box'
                Figs.make_box(csvs, var, stat, normalize_to_baseline=True)

            # Make plots of quantile flows for the gages
            else:
                if mode == 'ggo_cfs' or mode == 'ggo_diff':
                    for quantile in [0.1, 0.9]:
                        print 'Q{:.0f}0 Flows: '.format(10*(1-quantile)),

                        print 'Time series ',
                        #Figs.make_timeseries(csvs, var, stat, quantile=quantile, baseline=True)
                        Figs.make_timeseries_hexbin(csvs, var, stat, quantile=quantile, baseline=True)

                        #print 'Violin ',
                        #Figs.make_violin(csvs, var, stat, quantile=quantile)

                        #print 'Box'
                        Figs.make_box(csvs, var, stat, quantile=quantile, normalize_to_baseline=True)

    if Figs.stat_summary:
        Figs.ofp.close()
