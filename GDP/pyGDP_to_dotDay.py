__author__ = 'aleaf'
'''
Program to take WICCI csv files downloaded from the USGS-CIDA Geo Data Portal
and convert to PRMS .day format (read by the climate_hru module)

Input should be in single files, one for each scenario-time period-variable combination

Outputs PRMS .day files, one for each GCM-emissions scenario-time period-variable combination
- .day files contain information for a single variable with one column for each hru
- and one row for each timestep (e.g. day) of simulation

'''


import sys
sys.path.append('../Preprocessing')
import PRMSio
import pandas as pd


import os
import numpy as np
from collections import defaultdict

class gdpFiles:

    def __init__(self, f, suffix=''):

        print '{}\n'.format(f)

        self.name = os.path.split(f)[1]
        self.suffix = suffix
        self.scenario, self.gcm, self.par, self.period = self.name.split('-')
        self.realization = int(self.period.split('_')[0])
        self.conv_m = 1.0 # units conversion multiplier
        self.conv_c = 0.0 # units conversion constant
        self.conv = None

        # parse information from filename
        if '20c3m' in self.period:
            self.timeper = '1961-2000'
        elif 'early' in self.period:
            self.timeper = '2046-2065'
        elif 'late' in self.period:
            self.timeper = '2081-2100'
        else:
            raise Exception("Cannot parse time period from filename")

        self.dotDayfile = '{}.{}.{}.{}.{}_{}.day'.format(self.gcm, self.scenario, self.realization,
                                                      self.timeper, self.suffix, self.par)
        header = open(f).readlines()[0:3]
        units = header[2].split(',')[1]

        # units conversion
        if 'tm' in self.par and 'C' in units:
            self.conv = 'C to F'
            self.conv_m = (9/5.0) # C to F
            self.conv_c = 32.0
        elif 'pr' in self.par and 'mm' in units:
            self.conv = 'mm to inches'
            self.conv_m = (1/25.4) # mm to in.


    def convert_units(self, df):

        if 'pr' in self.par:
            df[df <= 5e-5] = 0 # set very small values to zero (caused by floating point errors in GDP)

        print 'converting units of {}'.format(self.conv)
        df = df * self.conv_m + self.conv_c

        return df


##################
## Main Program ##
##################



if __name__ == '__main__':

    csvdir = 'D:/ATLData/Fox-Wolf/GDP' # directory containing downloaded files from wicci
    datadir = 'D:/ATLData/Fox-Wolf/input' # directory for converted files
    suffix = 'fw' # written at end of filename
    overwriteOutput = True # T/F if output file already exists, overwrite it

    print "Getting list of csv files..."
    try:
        csvs = [os.path.join(csvdir, f) for f in os.listdir(csvdir) if f.lower().endswith('.csv')]
    except OSError:
        print "can't find directory with GDP files"


    # make the output directory if it doesn't exist
    if not os.path.isdir(datadir):
        os.makedirs(datadir)


    for f in csvs:

        # instantiate class for GDP file
        gdpf = gdpFiles(f, suffix)

        # read file into pandas dataframe
        df = pd.read_csv(f, skiprows=3, header=None, index_col=0, parse_dates=True)

        # convert the units
        df = gdpf.convert_units(df)

        # write back out to PRMS .day format
        dD = PRMSio.dotDay(df) # pass the dataframe to a dotDay object

        # set the header
        dD.header = ['created by pyGDP_to_dotDay.py\n{}     {}\n'.format(gdpf.par, dD.nhru) + 40*'#' + '\n']

        # save
        dD.write_output(os.path.join(datadir, gdpf.dotDayfile))

    print 'Done'