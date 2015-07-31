__author__ = 'aleaf'

import sys
sys.path.append('D:/ATLData/Documents/GitHub/GSFLOW_climate_utils')
import os
import numpy as np
import pandas as pd
import netCDF4
from PRMSio import netCDF4dataset
from GISio import get_proj4

path = 'D:/ATLData/BadRiver/swb/common_climate'

projection = '+proj=lcc +lat_1=25.0 +lat_2=60.0 +lat_0=42.5 +lon_0=-100.0 \
              +x_0=0.0 +y_0=0.0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs'

model_extent = 'D:/ATLData/BR/BadRiver/shps/GSFLOWnearfield.shp'
model_proj4 = get_proj4(model_extent)

ncdf = netCDF4dataset(proj4=projection, t0='1980-01-01 00:00:00')
template_file = os.path.join(path, os.listdir(path)[0])
ncdf.set_extent(template_file, model_extent=model_extent, reduce=2)

tmin_files = [os.path.join(path, f) for f in os.listdir(path) if 'tmin' in f and int(f[:4]) == 2000]
tmax_files = [os.path.join(path, f) for f in os.listdir(path) if 'tmax' in f and int(f[:4]) == 2000]
prcp_files = [os.path.join(path, f) for f in os.listdir(path) if 'prcp' in f and int(f[:4]) == 2000]

dftmin = ncdf.get_data(tmin_files, 'tmin', output_proj4=model_proj4)
dftmax = ncdf.get_data(tmax_files, 'tmax', output_proj4=model_proj4)
dfprcp = ncdf.get_data(prcp_files, 'prcp', output_proj4=model_proj4)

df = dftmin.join(dftmax['tmax'])
df = df.join(dfprcp['prcp'])

ncdf.write_to_dotData(df, 'BadRiver_daymet_{}.data'.format(2000))
#df.to_csv('BadRiver_daymet_{}-{}.csv'.format(year-2, year), index=False)

ncdf.print_stations(out_csv='Nearfield_stations.csv')
