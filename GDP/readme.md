#### Getting data from the USGS-CIDA Geo Data Portal using pyGDP:


**get_GDP_data.py**:  
  Python script used to fetch downscaled climate data from CIDA's Geo Data Portal (GDP).
  Acts as a driver for CIDA's **pyGDP** interface, which interacts with the GDP server to get the data.
  Includes restart functionality, for recovery from communications failures with the server.
  
**GDP_to_data.py**:  
  Takes files downloaded by get\_GDP_data.py and creates PRMS .data input files (used by PRMS ide module for interpolating tmin, tmax and precip across model domain using a set of stations).
  
  * generates a .data file per gcm-scenario-timeperiod (contains tmin, tmax, and precip)
  
 **GDP_to_dotDay.py**:  
  Takes files downloaded by get\_GDP_data.py and creates PRMS .day input files (used by PRMS climate_hru model for assigning tmin, tmax, and precip to each hru).
  
  * generates a .day file for each variable (tmin, tmax, precip), for each gcm-scenario-timeperiod