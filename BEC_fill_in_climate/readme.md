**Methods for filling in missing climate data for Black Earth Creek: (to allow for a longer model 'spinup' time)**

To simply copy for the first year for each discrete time period n times, and insert the copied years at the         simulation start:
 * run **GSFLOW_generate_synthetic.py** (generates new .data files with synthetic data inserted at the beginning) 
  
  Or, to fill in any data gaps with copied synthetic data, creating a single continuous time period:
 * run **PRMS_data_interp.py** (requires **pandas** and also **PRMS_data_interp_functions.py** to run)