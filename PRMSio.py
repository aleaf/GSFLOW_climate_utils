__author__ = 'aleaf'
'''
classes for reading and writing PRMS/GSFLOW input and output
'''

import os
import numpy as np
import pandas as pd
import datetime as dt
from collections import OrderedDict
import fiona
from shapely.geometry import shape, Point
import netCDF4
import pyproj
from GISio import get_proj4
from GISops import project


class parseFilenames(object):

    def __init__(self, filepath):

        self.f = filepath
        self.fname = os.path.split(self.f)[1]
        self.scenario, self.gcm, self.realization, self.timeperiod = self.fname.split('.')[:4]
        self.run = '.'.join((self.scenario, self.gcm, self.realization))
        self.suffix = self.f.split('.')[4].split('_')[0]

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


class netCDF4dataset:
    
    def __init__(self, x_col='x', y_col='y', time_col='time',
                 t0=0, time_units='d',
                 proj4=None):
        """
        Attributes
        ----------
        x_col : str
            name of variable in netCDF dataset containing x coordinates
        y_col : str
            name of variable in netCDF dataset containing y coordinates
        time_col : str
            name of variable in netCDF dataset containing timestamps
        t0 : str
            argument to pandas.to_datetime; adds an offset onto the time values.
        time_units : str
            argument to pandas.to_datetime specifying time units.
        X : array
            x-coordinates of points within model extent
        Y : array
            y-coordinates of points within model extent 
        """
        self.x_col = x_col
        self.y_col = y_col
        self.time_col = time_col
        self.time_units = time_units
        self.t0 = pd.to_datetime(t0)
        self._toffset = self.t0 - pd.to_datetime(0)
        
        # boolean arrays, of same length as x/y coordinates in dataset; 
        # True if coordinate within model bounding box
        self._yinds = np.array([])
        self._xinds = np.array([])
        
        self._within = np.array([]) # boolean array indicating which points are within model extent
        
        self.X = np.array([])
        self.Y = np.array([])
            
        self.proj4 = proj4
        
    def set_extent(self, ncfile, model_extent, model_extent_buffer=1000):
        
        f = netCDF4.Dataset(ncfile)
        
        # read geometry of model extent; project to coordinate system of netCDF data
        model_proj4 = get_proj4(model_extent)
        model_extent = shape(fiona.open(model_extent).next()['geometry'])
        model_extent = project(model_extent, model_proj4, self.proj4)
        model_extent_buff = model_extent.buffer(model_extent_buffer)

        X, Y = f.variables[self.x_col], f.variables[self.y_col]
        xinds = (X[:] > model_extent_buff.bounds[0]) & (X[:] < model_extent_buff.bounds[2])
        yinds = (Y[:] > model_extent_buff.bounds[1]) & (Y[:] < model_extent_buff.bounds[3])

        bbox_points_xy = np.reshape(np.meshgrid(X[xinds], Y[yinds]), (2, len(X[xinds]) * len(Y[yinds])))
        bbox_points = [Point(bbox_points_xy[0, i], bbox_points_xy[1, i]) for i in range(np.shape(bbox_points_xy)[1])]
        within = np.array([p.within(model_extent) for p in bbox_points])

        self._xinds = xinds # indices of points within model bounding box
        self._yinds = yinds
        self._within = within # boolean array indicating points within model extent
        self._allX = X # all x, y coordinates within netcdf dataset
        self._allY = Y
        self.X = bbox_points_xy[0, within]
        self.Y = bbox_points_xy[1, within]

    def get_data(self, ncfiles, var_col, dropna=True, 
                 output_proj4=None, datetime_output=True):
    
        if not isinstance(ncfiles, list):
            ncfiles = [ncfiles]
        
        if output_proj4 is not None:
            print 'reprojecting output coordinates to:\n{}\n'.format(output_proj4)
            pr1 = pyproj.Proj(self.proj4)
            pr2 = pyproj.Proj(output_proj4)
            X, Y = pyproj.transform(pr1, pr2, self.X, self.Y)
        else:
            X, Y = self.X, self.Y
        df = pd.DataFrame(columns=['point', self.x_col, self.y_col, var_col, self.time_col])
        for ncfile in ncfiles:
            print('\r{}'.format(ncfile)),
            f = netCDF4.Dataset(ncfile)
            var = f.variables[var_col]
            for i in range(var.shape[0]):
                # reshape variable values to 1-D array; cull to extent bbox
                var_rs = np.reshape(var[i, self._yinds, self._xinds], 
                                    (len(self._allX[self._xinds]) * len(self._allY[self._yinds])))
                # cull to actual extent
                var_rs = var_rs[self._within]
                time = f.variables[self.time_col][i]
                dft = pd.DataFrame({self.x_col: X,
                                    self.y_col: Y,
                                    self.time_col: time,
                                    var_col: var_rs})
                if datetime_output:
                    dft['time'] = pd.to_datetime(dft.time.values, unit=self.time_units) + self._toffset
                if dropna:
                    dft.dropna(axis=0, inplace=True)
                dft['point'] = range(len(dft))
                dft.index = pd.MultiIndex.from_product([np.unique(dft.time.values), dft.point.values])

                df = df.append(dft)
        print('\n')
        return df


class statvarFile(parseFilenames):

    def read2df(self):
        print '\nreading {}...'.format(self.f)

        # get header information
        self.nstats = int(open(self.f).readline())

        # strip off the end of lines, then strip trailing '1's on basin variables; attach any segment numbers to names
        self.statnames = [n.strip() for n in open(self.f).readlines()[:1+self.nstats]]
        self.statnames = [n.strip(' 1') if n.endswith(' 1') else n.replace(' ', '_') for n in self.statnames]

        # function to read dates from PRMS files
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')

        # read file into pandas dataframe (index_col is kludgy but works)
        df = pd.read_csv(self.f, delim_whitespace=True, header=None, skiprows=45, parse_dates=[[1, 2, 3, 4, 5, 6]],
                         date_parser=parse, index_col=0)

         # name the columns using the list of variables in the file header
        df.columns = self.statnames

        # drop the first column, which just contains consecutive integers and has nstatvar as its label
        df = df.drop(str(self.nstats), axis=1)

        return df


class dotDay:

    def __init__(self, df=pd.DataFrame()):

        self.df = df
        self.nhru = np.shape(df)[1]

    def transp(self, last_frost, first_frost, nhru):

        self.df_lf = last_frost
        self.df_ff = first_frost
        self.index = pd.date_range(dt.datetime(last_frost.index[0], 1, 1),
                                   dt.datetime(last_frost.index[-1], 12, 31))
        self.columns = np.arange(nhru) + 1
        self.header = []

        # convert frost dates to boolean series (loop seems clunky but not sure how to vectorize this)
        print '\n\nconverting growing season dates to boolean dataframe...'
        for year in self.df_lf.index:
            print year,
            l = [self.toBoolean(year, h) for h in (np.arange(nhru) + 1)] # using a list comprehension is MUCH faster than another loop
            transp_yr = pd.DataFrame(l).T # make a DataFrame for the year
            self.df = self.df.append(transp_yr)


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
        ndays = len(self.df)
        date_cols = [self.df.index.year,
                     self.df.index.month,
                     self.df.index.day,
                     0, 0, 0]
        names = ['Y', 'M', 'D', 'h', 'm', 's']
        for i in np.arange(6)+1:
            self.df.insert(0, names[-i], date_cols[-i])
        # add dataframe to output file
        self.df.to_csv(ofp, sep=' ', header=False, index=False, float_format='%.2f')

        ofp.close()

