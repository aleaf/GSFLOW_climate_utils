__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import datetime as dt
from collections import OrderedDict


class datafile:

    def __init__(self, datafile):

        self.f = datafile
        self.timeper = os.path.split(self.f)[1].split('.')[3]
        self.scenario = os.path.split(self.f)[1].split('.')[1]
        self.gcm = os.path.split(self.f)[1].split('.')[0]
        self.header = []
        self.Ncolumns = {}
        self.tmin = 0 # int, number of tmin columns
        self.tmax = 0 # int, number of tmax columns
        self.precip = 0 # int, number of precip columns

        # find first line of data; record header information
        temp = open(self.f).readlines()
        for line in temp:
            try:
                int(line.split()[0])
                break
            except:
                try:
                    ncols = int(line.strip().split()[1])
                    var = line.split()[0]
                    self.Ncolumns[var] = ncols
                    self.header.append(line.strip())
                    continue
                except:
                    self.header.append(line.strip())
                    continue

        self.tmin_start = 6 # assumes that the first 6 columns are for the date and time
        self.tmin_stop = 6
        tmin = False
        for line in self.header:
            if 'tmin' in line:
                self.tmin = int(line.strip().split()[1])
                tmin = True
                self.tmin_stop += self.tmin - 1
            elif 'tmax' in line:
                self.tmax = int(line.strip().split()[1])
                if not tmin:
                    self.tmin_start += self.tmax
                    self.tmin_stop += self.tmax
            elif 'prcp' in line:
                self.precip = int(line.strip().split()[1])
                if not tmin:
                    self.tmin_start += self.precip
                    self.tmin_stop += self.precip
        del temp


    def read2df(self):
        print '\nreading {}...'.format(self.f)

        # function to read dates from PRMS files
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')

        # read file into pandas dataframe (index_col is kludgy but works)
        df = pd.read_csv(self.f, delim_whitespace=True, dtype=None, header=None, skiprows=len(self.header),
                              parse_dates=[[0, 1, 2, 3, 4, 5]], date_parser=parse, index_col='0_1_2_3_4_5')

        return df


class dotDay:

    def __init__(self, last_frost, first_frost, nhru):

        self.df_lf = last_frost
        self.df_ff = first_frost
        self.index = pd.date_range(dt.datetime(last_frost.index[0], 1, 1),
                                   dt.datetime(last_frost.index[-1], 12, 31))
        self.columns = np.arange(nhru) + 1
        self.boolean_transp = pd.DataFrame()
        self.header = []

        # convert frost dates to boolean series (loop seems clunky but not sure how to vectorize this)
        print '\n\nconverting growing season dates to boolean dataframe...'
        for year in self.df_lf.index:
            print year,
            l = [self.toBoolean(year, h) for h in (np.arange(nhru) + 1)] # using a list comprehension is MUCH faster than another loop
            transp_yr = pd.DataFrame(l).T # make a DataFrame for the year
            self.boolean_transp = self.boolean_transp.append(transp_yr)


    def toBoolean(self, year, hru):
        '''
        takes a year, last and first frost date,
        and returns a daily series of zeros (no transpiration), and ones (transpiration)
        '''
        last_frost, first_frost = self.df_lf.ix[year, hru], self.df_ff.ix[year, hru]

        # make datetime64 objects for start/end of year and growing season
        Jan1, Dec31 = pd.Timestamp(dt.datetime(year, 1, 1)), pd.Timestamp(dt.datetime(year, 12, 31))
        last_frost_date, first_frost_date = Jan1 + np.timedelta64(last_frost, 'D'), Jan1 + np.timedelta64(first_frost, 'D')

        # make index for each period
        lateWinter = pd.date_range(start=Jan1, end=last_frost_date)
        summer = pd.date_range(start=last_frost_date, end=first_frost_date)[1:-1] # pandas includes end dates in slice
        earlyWinter = pd.date_range(start=first_frost_date, end=Dec31)

        # put them together into a boolean series (n days x 1 hru)
        days = pd.Series(data=0, index=lateWinter)
        days = days.append(pd.Series(data=1, index=summer))
        days = days.append(pd.Series(data=0, index=earlyWinter))

        return days


    def write_output(self, outfile):

        ofp = open(outfile, 'w')
        print '\nwriting output to %s\n' %(outfile)

        for lines in self.header:
            ofp.write(lines)

        # insert columns for PRMS date
        ndays = len(self.boolean_transp)
        date_cols = [self.boolean_transp.index.year,
                     self.boolean_transp.index.month,
                     self.boolean_transp.index.day,
                     0, 0, 0]
        names = ['Y', 'M', 'D', 'h', 'm', 's']
        for i in np.arange(6)+1:
            self.boolean_transp.insert(0, names[-i], date_cols[-i])
        # add dataframe to output file
        self.boolean_transp.to_csv(ofp, sep=' ', header=False, index=False)

        ofp.close()

