'''

generates a batch of control files for multiple GSFLOW runs, from list of input files
for naming to work correctly, .data or .day filesnames should
    - start with GCM.Scenario.Realization.TimePeriod
    - end with .day or .data

'''
import os
import datetime

# input
datadir = 'D:/ATLData/Fox-Wolf/input' # directory with PRMS data or .day files
controltemplate = 'D:/ATLData/Fox-Wolf/fox_climate_hru_future.control'
suffix = 'fw' # prefix indicating the model (e.g. for csv output file -> <prefix>_out.csv )
model_mode = "PRMS" # PRMS or GSFLOW; mode to write in control file

# output
controldir = 'D:/ATLData/Fox-Wolf/control' # where to save .control files
outputdir = 'D:/ATLData/Fox-Wolf/output' # subfolder in slave dir where output files will be saved during individual scenario runs
paramsdir = 'params' # subfolder in slave dir where params files will be stored during individual scenario runs
#preprocdir='preprocessing' # folder where tmin.day files are stored (only for WRITE_CLIMATE mode)

climate_method = 'climate_hru' # 'ide' or 'climate_hru' # if ide, will write name of .data file; if 'climate_hru', will write .day files
preproc = False # T/F whether or not control files are for 'preprocessing' run

now = datetime.datetime.now()

# make an output folder if there isn't one already
if not os.path.isdir(controldir):
    os.makedirs(controldir)

# generate list of unique basenames (GCM.Scenario.Realization.TimePeriod);
# append suffix to them
basenames = list(set(['.'.join(f.split('.')[:4]) + '.{}'.format(suffix)
                      for f in os.listdir(datadir) if f.endswith('.data') or f.endswith('.day')]))

# check to make sure that all of the files are in the input folder
nfiles_per_basename = []
for b in basenames:
    files = [f for f in os.listdir(datadir) if b in f]
    nfiles_per_basename.append(len(files))
    if len(set(nfiles_per_basename)) > 1:
        raise Exception("Not all basenames have same number of files. Check {} folder.".format(datadir))

controldata = open(controltemplate, 'r').readlines()

print "writing GSFLOW control files..."

count = 0
# iterate through basenames; use them to come up with input files names for PRMS control file
for b in basenames:
    count += 1

    control_file_name = b + '.control'
    print control_file_name

    GCM, scenario, realization, timeper = b.split('.')[:4]

    ofp = open(os.path.join(controldir, control_file_name), 'w')
    ofp.write('Fox-Wolf PRMS - GCM: %s, Scenario: %s, Realization: %s, %s, created by gsflow_control_generator.py on %s/%s/%s\n'
              %(GCM, scenario, realization, timeper, now.month, now.day, now.year))

    for i in range(len(controldata))[1:]:
        line = controldata[i]
        if i < 3:
            ofp.write(line)
        else:
            if "csv_output_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir, b + '_out.csv\n'))
            elif climate_method == 'ide' and "data_file" in controldata[i-3]:
                ofp.write(os.path.join(datadir, b + '.data')+'\n')
            elif "end_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[1]+'\n')
            elif "model_mode" in controldata[i-3]:
                ofp.write(model_mode+'\n')
            elif "model_output_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir, b + '_{}.out\n'.format(model_mode)))
            elif "start_time" in controldata[i-3]:
                ofp.write(timeper.split('-')[0]+'\n')
            elif "stat_var_file" in controldata[i-3]:
                ofp.write(os.path.join(outputdir, b + '.statvar.dat\n'))
            elif preproc and "param_file" in controldata[i-4]:
                ofp.write(os.path.join(paramsdir, b + '_preprocess.params\n'))
            elif 'ani_output_file' in controldata[i-3]:
                ofp.write(os.path.join(outputdir, b + '.ani.dat\n'))
                
            # these '_day' entries are only needed if using climate_hru module
            elif climate_method == 'climate_hru' and 'tmax_day' in controldata[i-3]:
                ofp.write(os.path.join(datadir, b + '_tmax.day\n'))
            elif climate_method == 'climate_hru' and 'tmin_day' in controldata[i-3]:
                ofp.write(os.path.join(datadir, b + '_tmin.day\n'))
            elif climate_method == 'climate_hru' and 'precip_day' in controldata[i-3]:
                ofp.write(os.path.join(datadir, b + '_prcp.day\n'))
            elif "transp_day" in controldata[i-3]:
                ofp.write(os.path.join(datadir, b + '_transp.day\n'))
            else:
                ofp.write(line)
    ofp.close()
print "Done!, wrote %s files" %(count)     