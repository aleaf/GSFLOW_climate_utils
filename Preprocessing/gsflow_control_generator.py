# generates a batch of control files for multiple GSFLOW runs, from list of input files
# data filesnames should follow convention of GCM.Scenario.Realization.TimePeriod.**.data,  for naming to work correctly

import os
import datetime

# input
datadir = 'D:/ATLData/Fox-Wolf/data' # directory with PRMS data files
controltemplate = 'gsflow_pest.control'
model_mode = "GSFLOW" # PRMS or GSFLOW; mode to write in control file

# output
controldir = 'D:/ATLData/Fox-Wolf/data/control' # where to save .control files
outputdir = 'output' # subfolder in slave dir where output files will be saved during individual scenario runs
paramsdir = 'params' # subfolder in slave dir where params files will be stored during individual scenario runs
#preprocdir='preprocessing' # folder where tmin.day files are stored (only for WRITE_CLIMATE mode)

WRITE_CLIMATE = False # T/F T to create input files for running GSFLOW/PRMS in WRITE_CLIMATE (preprocessing) mode

now = datetime.datetime.now()

print "Getting list of data files..."
allfiles = os.listdir(datadir)

datas=[f for f in allfiles if f.lower().endswith('.data')]

controldata=open(controltemplate,'r').readlines()

print "writing GSFLOW control files..."

count=0
for files in datas:
    count+=1
    basename=files[:-5]
    print basename+'.control'
    fname=files.split('.')
    GCM,scenario,realization,timeper=fname[0:4]
    ofp=open(os.path.join(controldir,basename+'.control'),'w')
    ofp.write('BEC GSFLOW - GCM: %s, Scenario: %s, Realization: %s, %s, created by gsflow_control_generator.py on %s/%s/%s\n' %(GCM,scenario,realization,timeper,now.month,now.day,now.year))
    for i in range(len(controldata))[1:]:
        line=controldata[i]
        if i<3:
            ofp.write(line)
        else:
            if "csv_output_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir,basename+'.bec_out.csv\n'))
            elif "data_file" in controldata[i-3]:
                ofp.write(os.path.join(datadir,files)+'\n')
            elif "end_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[1]+'\n')
            elif "model_mode" in controldata[i-3]:
                ofp.write(model_mode+'\n')
            elif "model_output_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir,basename+'.BEC_GSFLOW.out\n'))
            elif "start_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[0]+'\n')
            elif "stat_var_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir,basename+'.statvar.dat\n'))
            elif "param_file" in controldata[i-4]:
                ofp.write(os.path.join(paramsdir,basename+'_preprocess.params\n'))
                
            elif WRITE_CLIMATE:
                if "transp_module" in controldata[i-3]:
                    ofp.write('climate_hru\n')
                elif "transp_day" in controldata[i-3]:
                    ofp.write(os.path.join(datadir,files[:-8]+'transp.day\n'))
                else:
                    ofp.write(line)
            else:
                ofp.write(line)
    ofp.close()
print "Done!, wrote %s files" %(count)     