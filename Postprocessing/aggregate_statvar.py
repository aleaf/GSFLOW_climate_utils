__author__ = 'aleaf'
'''
program for aggregating results from numerous runs into csv files by variable
'''
import os
import pandas as pd
import sys
sys.path.append('..')
from PRMSio import statvarFile

PRMSresults_folder = 'D:/ATLData/Fox-Wolf/output'
output_folder = 'D:/ATLData/Fox-Wolf/statvar'

if not os.path.isdir(output_folder):
    os.makedirs(output_folder)

statvarfiles = sorted([os.path.join(PRMSresults_folder, f) for f in os.listdir(PRMSresults_folder)
                       if 'statvar' in f and f.endswith('.dat')])

# read all statvar files into one giant dataframe
scenarios = list(set([f.split('.')[1] for f in statvarfiles if '20c3m' not in f]))
df_all = pd.DataFrame()
for f in statvarfiles:

    # instantiate statvar file object
    sv = statvarFile(f)

    # read contents of statvar file to pandas dataframe
    df = sv.read2df()

    # assign a run identifier to dataframe columns
    # append 20th century runs for each scenario (makes for a larger dataframe, but simplifies pivoting)
    if '20c3m' in sv.run:
        for s in scenarios:
            s20th = sv.run.replace('20c3m', s)
            df['run'] = s20th

            # add to master dataframe
            df_all = df_all.append(df)
    else:
        df['run'] = sv.run
        df_all = df_all.append(df)

# for each variable...
print '\npivoting to variables and saving...'
variables = [c for c in df_all.columns if c != 'run']

for var in df_all.columns:
    print var,
    # select data for variable
    df_all['Date'] = df_all.index
    dfv = df_all[['Date', var, 'run']]

    # pivoted dataframe of single variable, with one column for each gcm-scenario-realization
    dfv = dfv.pivot(index='Date', columns='run', values=var)

    # save out a csv for each scenario in dfv
    for s in scenarios:

        # select only the dataframe columns for the scenario
        scen_columns = [c for c in dfv.columns if s in c]
        dfvs = dfv[scen_columns]

        # remove scenario info from columns (already in filename)
        dfvs.columns = [c.replace('{}.'.format(s), '') for c in dfvs.columns]

    outcsv = os.path.join(output_folder, '{}.{}.csv'.format(var, s))
    dfvs.to_csv(outcsv, index_label='Date')

print '\n\nDone'



