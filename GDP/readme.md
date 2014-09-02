#### Getting data from the USGS-CIDA Geo Data Portal using pyGDP:


**get_GDP_data.py**:  
  Python script used to fetch downscaled climate data from CIDA's Geo Data Portal (GDP).
  Acts as a driver for CIDA's **pyGDP** interface, which interacts with the GDP server to get the data.
  Includes restart functionality, for recovery from communications failures with the server.
  
**GDP_to_data.py**:  
  Takes files downloaded by get\_GDP_data.py and creates PRMS .data input files.