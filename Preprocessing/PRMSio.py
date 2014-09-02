__author__ = 'aleaf'

import os
import pandas as pd
import datetime as dt


class datafile:

    def __init__(self, datafile):

        self.f = datafile
        self.timeper = os.path.split(self.f)[1].split('.')[3]
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
        print self.f

        # function to read dates from PRMS files
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')

        # read file into pandas dataframe (index_col is kludgy but works)
        df = pd.read_csv(self.f, delim_whitespace=True, dtype=None, header=None, skiprows=len(self.header),
                              parse_dates=[[0, 1, 2, 3, 4, 5]], date_parser=parse, index_col='0_1_2_3_4_5')

        return df


class dotDay:

    def __init(self):
        j=2

