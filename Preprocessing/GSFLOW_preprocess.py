# program for batch preprocessing of GSFLOW to determine growing season
# produces a preprocess.params file for each input .data file

import os
import datetime

# inputs:
datadir='continuous_input' # directory of .data input files
preprocessing_dir='preprocessing' # directory to save preprocessing files

mode='WRITE_CLIMATE' # preprocess or WRITE_CLIMATE

# these next 3 files should all go in the preprocessing_dir
paramsfile='bec_prms.params' # params file for all runs

if mode=='WRITE_CLIMATE':
    preproc_control='bec_prms_preproc_write_climate.control' # template control file
    prms_exec='gsflow_develop_21Jun12.exe'
    commandline='%s %s'

else:
    preproc_control='bec_prms_preproc.control'
    prms_exec='prms2010_Fmay21.exe' # PRMS executable with preprocessing capability
    commandline='%s %s -preprocess'


now=datetime.datetime.now()

print "Getting list of data files..."
allfiles=os.listdir(datadir)

datas=[f for f in allfiles if f.lower().endswith('.data')]

controldata=open(os.path.join(preprocessing_dir,preproc_control),'r').readlines()

print "creating preprocessing control files and running PRMS in preprocess mode..."
count=0
os.chdir(preprocessing_dir)
for files in datas:
    count+=1
    basename=files[:-5]
    print basename+'.control'
    fname=files.split('.')
    GCM,scenario,realization,timeper=fname[0:4]
    ofp=open(basename+'.control','w')
    for i in range(len(controldata))[1:]:
        line=controldata[i]
        if i<3:
            ofp.write(line)
        else:
            if "data_file" in controldata[i-3]:
                ofp.write(os.path.join('..',datadir,files)+'\n')
            elif "end_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[1]+'\n')
            elif "start_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[0]+'\n')
            elif "param_file" in controldata[i-3]:
                ofp.write(paramsfile+'\n')
            # Note: these are only needed to satisfy PRMS;
            # preproceeing mode doesn't output anything useful to these files
            elif "param_file" in controldata[i-4]:
                ofp.write(basename+'_preprocess.params\n')
            elif "stat_var_file" in controldata[i-3]:
                ofp.write('BEC_statvar.dat\n')
            else:
                ofp.write(line)
    ofp.close()
    os.system(commandline %(prms_exec,basename+'.control'))
    if mode=='WRITE_CLIMATE':
        os.rename('tmin.day',basename+'_tmin.day')
    if mode<>'WRITE_CLIMATE':
        os.rename('bec_prms.params_preprocess.params',basename+'_preprocess.params')
    os.remove(basename+'.control')
    percentdone=100*count/len(datas)
    print '%s %% done' %(round(percentdone,1))
os.remove('BEC_statvar.dat')
os.remove('BEC_GSFLOW.out')
#os.chdir('..')
print "Done!, wrote %s files" %(count)