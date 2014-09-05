
  
    
    
#### Workflow for batching multiple runs for multiple future climate scenarios:  

#####using simple python script
(i.e. for PRMS, which runs fast)

* see **runner_fw.py** for example
	
#####**using HTCondor:**

4. Setup a generic run folder for HTCondor which includes:

 	* a subfolder 'input' with all of the .data files (one for each run) and transp.day files (if using WRITE_CLIMATE module)
 	* a subfolder 'control' with all of the .control files (one for each run)
 	* a subfolder 'params' with all of the preprocess.params files (one for each run); also should contain a generic .params file that has information for all runs
 	* in root directory, all of the MODFLOW files
 	* runner.py in the root directory

5. zip up the generic run folder; also zip up a copy of Python27

6. setup HTCondor worker.sub file so that
 	* the generic run folder, Python27, and unzip.exe are sent
 	* queue= number of model runs
 	* request_disk is set high enough for recording daily input to heads
 	* arguments= $(Process)     this passes the job number as an argument
 	* transfer_output = <generic run folder name>\BEC_run_$(Process).zip
 	* transfer_output_remaps ="BEC_run_$(Process).zip = results\BEC_run_$(Process).zip"     this neatly puts all of the zips in a results folder

7. **runner.py:** 
 	* takes the job number then as an argument (so adjust HTCondor batch file to launch runner.py with one argument)
 	* runner.py also creates a MODFLOW output control file on the fly, so that oc is correct regardless of what time period is being run
 	* GSFLOW is then launched, followed by MOD2SMP, which extracts head data at pre-specified locations
 	* finally, everything is zipped up
 	* outputs 'BEC_run_$(Process).zip'      (e.g. BEC_run_0 for the first job) 

8. run HTCondor  
9. Postprocess the output (see readme in Postprocessing folder)

