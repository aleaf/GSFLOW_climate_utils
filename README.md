pyGDP
=====
BEC_GDP.py
  Python script used to fetch downscaled climate data from CIDA's Geo Data Portal (GDP)
  Acts as a driver for CIDA's pyGDP script, which interacts with the server to get the data
  Includes restart functionality, for recovery from communications failures with the server.
  
GDP_to_data.py
  Takes files downloaded by BEC_GPD.py and creates PRMS input files
