pyGDP
=====
BEC_GDP.py:
  Python script used to fetch downscaled climate data from CIDA's Geo Data Portal (GDP).
  Acts as a driver for CIDA's pyGDP script, which interacts with the GDP server to get the data.
  Includes restart functionality, for recovery from communications failures with the server.
  
GDP_to_data.py:
  Takes files downloaded by BEC_GPD.py and creates PRMS .data input files.
  

#### Workflow for setting up GSFLOW input for multiple future climate scenarios:  

run pyGDP_to_data.py to generate .data files for PRMS input

if adding synthetic data for spin-up:


     To simply copy for the first year for each discrete time period n times, and insert the copied years at the simulation start:

          -run GSFLOW_generate_synthetic.py (generates new .data files with synthetic data inserted at the beginning)     
     Or, to fill in any data gaps with copied synthetic data, creating a single continuous time period:
          - run PRMS_data_interp.py (requires pandas and also PRMS_data_interp_functions.py to run)


run GSFLOW_preprocess.py (generates a preprocess.params file with information on the growing season, for each model run)
     for variable growing season preprocessing (WRITE_CLIMATE mode):
          - made another template control file in the preprocessing directory that has 'WRITE_CLIMATE' instead of PRMS for model mode
          - run GSFLOW_preprocess.py in WRITE_CLIMATE mode, to generate a set of tmin.day files
          - next step is to generate a control file (one for each run) that references its respective tmin.day file (added tmin_day section to gsflow_pest.control- the template control file for gsflow_control_generator.py)
          - ran GSFLOW_control_generator.py with line added to put in tmin_day file reference for respective tmin_day file; saved control files to "control_continuous" folder
          - ran transp_preproc_driver.py- outputs *.year files to "frost" folder

run GSFLOW_control_generator.py (generates a control file for each model run)

setup a generic run folder for HTCondor which includes:
     - a folder 'input' with all of the .data files (one for each run) and transp.day files (if using WRITE_CLIMATE module)
     - a folder 'control' with all of the .control files (one for each run)
     - a folder 'params' with all of the preprocess.params files (one for each run); also should contain a generic .params file that has information for all runs
     - in root directory, all of the MODFLOW files
     - runner.py in the root directory

zip up the generic run folder; also zip up a copy of Python27

setup HTCondor worker.sub file so that
     -the generic run folder, Python, and unzip.exe are sent
     -queue= number of model runs
     -request_disk is set high enough for recording daily input to heads
     -arguments= $(Process)     this passes the job number as an argument
          -runner.py  
               -takes the job number then as an argument (so adjust HTCondor batch file to launch runner.py with one argument)
               - runner.py also creates a MODFLOW output control file on the fly, so that oc is correct regardless of what time period is being run
               - GSFLOW is then launched, followed by MOD2SMP, which extracts head data at pre-specified locations
               -finally, everything is zipped up
               -outputs 'BEC_run_$(Process).zip'      (e.g. BEC_run_0 for the first job)
     -transfer_output = <generic run folder name>\BEC_run_$(Process).zip
     -transfer_output_remaps ="BEC_run_$(Process).zip = results\BEC_run_$(Process).zip"     this neatly puts all of the zips in a results folder

run HTCondor

setup a folder for each type of result (see instructions) then run aggregate_GSFLOW.py on results folder     (aggregates results by variable.scenario (one csv for each, which a column of daily results for each GCM))

run GSFLOW_plots.py to generate plots. Both aggregate_GSFLOW and GSFLOW_plots use functions stored in GSFLOW_utils.py. GSFLOW_utils requires the pandas module of Python to run.
