#### Setting up input files for future climate scenarios:

GSFLOW/PRMS requires information on the growing season, defined as the time period between the last killing frost of the year, and first killing frost of the year. Two options for developing growing season input from daily minimum temperature data:

**use transp_preproc.py to write a set of transp.day files directly**  
(transp.day files tell GSFLOW/PRMS whether transpiration is on or off in each hru, on each day) 

 * requires a set of GSFLOW/PRMS .data files (i.e. from **pyGDP_to_data.py**) 
 * the growing season can be determined by hru, or globally for the whole domain (the latter option needs a bit of work to be updated to the new object-oriented framework of transp_preproc.py)
 * **transp_preproc.py** can also make a timeseries plot of the growing season length over time, for the suite of gcm-scenario combinations
 * outputs transp.day files, one for each gcm-scenario-time period combination
 * the transp.day files are included with the .data files in the input folder, and specified in the GSFLOW/PRMS control file


**use GSFLOW/PRMS to calculate the growing season via GSFLOW_preprocess.py**  
  (generates a preprocess.params file with information on the growing season, for each model run)

 * make another template control file in the preprocessing directory that has 'WRITE_CLIMATE' instead of PRMS for 'model mode'
 * run **GSFLOW_preprocess.py** in WRITE_CLIMATE mode, to generate a set of tmin.day files  
 * next step is to generate a control file (one for each run) that references its respective tmin.day file (added tmin_day section to gsflow_pest.control- the template control file for gsflow_control_generator.py)  
 * ran **GSFLOW_control_generator.py** with line added to put in tmin_day file reference for respective tmin_day file; saved control files to "control_continuous" folder  
 * ran **transp_preproc_driver.py**- outputs *.year files to "frost" folder