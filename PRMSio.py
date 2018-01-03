__author__ = 'aleaf'
'''
classes for reading and writing PRMS/GSFLOW input and output
'''

import os
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from collections import OrderedDict
import fiona
from shapely.geometry import shape, Point
import netCDF4
import pyproj
from GISio import get_proj4
from GISops import project


def prms_date(df):
    date_cols = [df.index.year,
                 df.index.month,
                 df.index.day,
                         0, 0, 0]
    names = ['Y', 'M', 'D', 'h', 'm', 's']
    for i in np.arange(6)+1:
        df.insert(0, names[-i], date_cols[-i])


class parseFilenames(object):

    def __init__(self, filepath):

        self.f = filepath
        self.fname = os.path.split(self.f)[1]
        self.scenario, self.gcm, self.realization, self.timeperiod = self.fname.split('.')[:4]
        self.run = '.'.join((self.scenario, self.gcm, self.realization))
        self.suffix = self.f.split('.')[4].split('_')[0]

class datafile:

    def __init__(self, datafile, parse_timeper=None, parse_scenario=None, parse_gcm=None):

        self.f = datafile
        self.timeper = None #os.path.split(self.f)[1].split('.')[3]
        self.scenario = None #os.path.split(self.f)[1].split('.')[1]
        self.gcm = None #os.path.split(self.f)[1].split('.')[0]

        for func, attr in list({parse_timeper: 'timeper',
                           parse_scenario: 'scenario',
                           parse_gcm: 'gcm'}.items()):
            if func is not None:
                self.__dict__[attr] = func(datafile)

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
        print('\nreading {}...'.format(self.f))

        # function to read dates from PRMS files
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')

        # read file into pandas dataframe (index_col is kludgy but works)
        df = pd.read_csv(self.f, delim_whitespace=True, dtype=None, header=None, skiprows=len(self.header),
                              parse_dates=[[0, 1, 2, 3, 4, 5]], date_parser=parse, index_col='0_1_2_3_4_5')

        return df


class netCDF4dataset:
    
    def __init__(self, x_col='x', y_col='y', time_col='time',
                 t0=0, time_units='d',
                 proj4=None, output_proj4=None):
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
        self.outputX = np.array([])
        self.outputY = np.array([])
            
        self.proj4 = proj4
        self.output_proj4 = output_proj4
        self.reproject_output(output_proj4=output_proj4)
        
    def set_extent(self, ncfile, model_extent, model_extent_buffer=1000, reduce=1):

        print('setting extent to {}\n\tusing points in {}...'.format(model_extent, ncfile))
        f = netCDF4.Dataset(ncfile)
        
        # read geometry of model extent; project to coordinate system of netCDF data
        model_proj4 = get_proj4(model_extent)
        model_extent = shape(fiona.open(model_extent).next()['geometry'])
        model_extent = project(model_extent, model_proj4, self.proj4)
        model_extent_buff = model_extent.buffer(model_extent_buffer)

        # get x and y locations
        X, Y = f.variables[self.x_col], f.variables[self.y_col]

        # build a mask for data; exclude points outside the model bounding box
        xreduce = np.array([False] * len(X))
        yreduce = np.array([False] * len(Y))
        xreduce[::reduce] = True
        yreduce[::reduce] = True

        xinds = (X[:] > model_extent_buff.bounds[0]) & (X[:] < model_extent_buff.bounds[2]) & xreduce
        yinds = (Y[:] > model_extent_buff.bounds[1]) & (Y[:] < model_extent_buff.bounds[3]) & yreduce

        # get a variable in the file being used to set the extent
        # (must have t, x, y dimensions)
        varname = [k for k, v in list(f.variables.items()) if len(v.shape) == 3][0]
        var = f.variables[varname]
        var_rs = np.reshape(var[0, yinds, xinds], (len(X[xinds]) * len(Y[yinds])))

        bbox_points_xy = np.reshape(np.meshgrid(X[xinds], Y[yinds]), (2, len(X[xinds]) * len(Y[yinds])))
        bbox_points = [Point(bbox_points_xy[0, i], bbox_points_xy[1, i]) for i in range(np.shape(bbox_points_xy)[1])]

        # create boolean index of whether points are in model extent and not masked in dayment
        within = np.array([p.within(model_extent)
                           if not var_rs.mask[i]
                           else False for i, p in enumerate(bbox_points)])


        self._xinds = xinds # indices of points within model bounding box
        self._yinds = yinds
        self._allX = X # all x, y coordinates within netcdf dataset
        self._allY = Y
        self._within = within # boolean array indicating points within model extent
        self.X = bbox_points_xy[0, within]
        self.Y = bbox_points_xy[1, within]

    def get_data(self, ncfiles, var_col,
                 output_proj4=None, datetime_output=True, fill_leap_years=True):

        self.datetime_output = datetime_output
        self.fill_leap_years = fill_leap_years

        if not isinstance(ncfiles, list):
            ncfiles = [ncfiles]

        self.reproject_output(output_proj4=output_proj4)

        df = pd.DataFrame(columns=['point', self.x_col, self.y_col, var_col, self.time_col])
        for ncfile in ncfiles:
            print(('\r{}'.format(ncfile)), end=' ')
            f = netCDF4.Dataset(ncfile)
            var = f.variables[var_col]
            for i in range(var.shape[0]):

                time = f.variables[self.time_col][i]
                datetime = pd.to_datetime(time, unit=self.time_units) + self._toffset

                # reshape variable values to 1-D array; cull to extent bbox
                var_rs = self._get_values(var, i)

                # make dataframe of values for the current timestamp
                dft = self._build_timestamp_dataframe(var_rs, var_col, time)
                df = df.append(dft)

                # check if current timestamp is 12/30 on a leap year
                if self.fill_leap_years and self.datetime_output \
                                and calendar.isleap(datetime.year) \
                                and datetime.month == 12 \
                                and datetime.day == 30:

                    # make another dataframe for 12/31, which is missing
                    # for now, simply copy 12/30 (for daymet, 1/1 is in another file)
                    dft = self._build_timestamp_dataframe(var_rs, var_col, time+1)
                    df = df.append(dft)
        self.df = df
        if self.fill_leap_years:
            df = self._compute_leapyear_lastdays(df, var_col)
        return df

    def _build_timestamp_dataframe(self, var_rs, var_col, time):

        dft = pd.DataFrame({self.x_col: self.outputX,
                            self.y_col: self.outputY,
                            self.time_col: time,
                            var_col: var_rs})

        if self.datetime_output:
            dft['time'] = pd.to_datetime(dft.time.values, unit=self.time_units) + self._toffset
        dft['point'] = list(range(len(dft)))
        dft.index = pd.MultiIndex.from_product([np.unique(dft.time.values), dft.point.values])

        return dft

    def _get_values(self, var, i):

        # reshape variable values to 1-D array; cull to extent bbox
        var_rs = np.reshape(var[i, self._yinds, self._xinds],
                            (len(self._allX[self._xinds]) * len(self._allY[self._yinds])))
        # cull to actual extent (this excludes masked points in daymet; see set_extent()
        return var_rs[self._within].copy()

    def _compute_leapyear_lastdays(self, df, var_col):

        # get leap years in dataframe, as long as they are followed by a Jan 1
        # those that are the last year will be left as-is (filled by copying Dec 30)
        leap_years = np.array([y for y in df.time.dt.year.unique() if calendar.isleap(y) and
                                          y + 1 in list(df.time.dt.year.unique())])
        next_years = leap_years + 1 # for Jan 1

        leap_year_inds = (df.time.dt.month == 12) & (df.time.dt.year.isin(list(leap_years)))
        next_year_inds = (df.time.dt.month == 1) & (df.time.dt.year.isin(list(next_years)))

        # for all leap years in dataframe, average 12/30 and 1/1
        df.loc[(df.time.dt.day == 31) & leap_year_inds, var_col] = \
            (df.ix[(df.time.dt.day == 30) & leap_year_inds, var_col].values +
             df.ix[(df.time.dt.day == 1) & next_year_inds, var_col].values) / 2.0
        return df

    def print_stations(self, out_csv='stations.csv',
                       **kwargs):

        df = pd.DataFrame({'station': np.arange(len(self.outputX)) + 1,
                           self.x_col: self.outputX,
                           self.y_col: self.outputY})
        print('writing station coordinates to {}'.format(out_csv))
        df.to_csv(out_csv, index=False, **kwargs)

    def reproject_output(self, output_proj4=None):

        self.output_proj4 = output_proj4
        if len(self.outputX) == 0 and output_proj4 is not None:
            print('reprojecting output coordinates to:\n{}\n'.format(output_proj4))
            pr1 = pyproj.Proj(self.proj4)
            pr2 = pyproj.Proj(self.output_proj4)
            self.outputX, self.outputY = pyproj.transform(pr1, pr2, self.X, self.Y)
        elif output_proj4 is None:
            self.outputX, self.outputY = self.X, self.Y
        else:
            pass


    def write_to_dotData(self, df, outfile='climate.data'):

        print('writing {}'.format(outfile))
        df['point2'] = df.point + df.point.max() + 1
        df['point3'] = df.point + df.point2.max() + 1

        data = pd.DataFrame(index=df.time.unique())
        for col, var in list({'point': 'tmin', 'point2': 'tmax', 'point3': 'prcp'}.items()):
            pivot = df.pivot(index='time', columns=col, values=var).copy()
            data = data.join(pivot)

        data.sort_index(axis=1, inplace=True)
        prms_date(data)

        f = open(outfile, 'w')
        npoints = len(np.unique(df.point))
        f.write('#Data file written by GSFLOW_climate_utils\n')
        f.write('tmin {0:.0f}\ntmax {0:.0f}\nprecip {0:.0f}\n'.format(npoints))
        f.write('#'*40 + '\n')
        data.to_csv(f, sep=' ', header=None, index=False)


class statvarFile(parseFilenames):

    def read2df(self):
        print('\nreading {}...'.format(self.f))

        # get header information
        self.nstats = int(open(self.f).readline())

        # strip off the end of lines, then strip trailing '1's on basin variables; attach any segment numbers to names
        self.statnames = [n.strip() for n in open(self.f).readlines()[:1+self.nstats]]
        self.statnames = [n.strip(' 1') if n.endswith(' 1') else n.replace(' ', '_') for n in self.statnames]

        # function to read dates from PRMS files
        parse = lambda x: dt.datetime.strptime(x, '%Y %m %d %H %M %S')

        # read file into pandas dataframe (index_col is kludgy but works)
        df = pd.read_csv(self.f, delim_whitespace=True, header=None, skiprows=self.nstats + 1, parse_dates=[[1, 2, 3, 4, 5, 6]],
                         date_parser=parse, index_col=0)
        df.index.name = 'Datetime'

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
        print('\n\nconverting growing season dates to boolean dataframe...')
        for year in self.df_lf.index:
            print(year, end=' ')
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
        print('\nwriting output to %s\n' %(outfile))

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

