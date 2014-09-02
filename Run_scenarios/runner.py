# program to run an instance of a GSFLOW model using HTCondor
# in Condor batch file, call this file with an argument that is tied to the job number
# i.e. in worker.bat, python runner.py %1; then in SUB file, arguments = $(Process)

import os
import sys
import datetime
import zipfile
from collections import defaultdict

OCmode='periods' # 'continuous' or 'periods'
OCperiods=['1961-2000','2046-2065','2081-2100'] # list of periods for which to print output ['startyear-endyear',...]

# make a list of GSFLOW control files for each run; assign to consecutive numbers starting at 0
allfiles=os.listdir('control')
controlfiles=[f for f in allfiles if f.lower().endswith('.control')]
controlfiles=sorted(controlfiles)

# get unique job number from Condor
jobnumber=int(sys.argv[1])

# assign control file based on job number
file2run=controlfiles[jobnumber].strip()
print "\nsetting up to run %s" %(file2run)

# get number of timesteps from filename (assumes daily timesteps)
fname=file2run.split('.')
GCM,scenario,realization,timeper=fname[0:4]
startyr,endyr=map(int,timeper.split('-'))
timesteps=datetime.date(endyr,12,31)-datetime.date(startyr,1,1)
timesteps=timesteps.days

# create output control for MODFLOW, based on timeperiod
SP=defaultdict()
SP[1]=range(2) # number of timesteps in stress period 1
if OCmode=='continuous':
    SP[2]=range(timesteps+1)[1:] # number of timesteps in sp 2
    header='#OC generated for years %s\n' %(timeper)
    
# create output control that will only save for desired periods (above)
elif OCmode=='periods':
    header='#OC generated for years %s\n' %(', '.join(OCperiods))
    OCstarts=[]
    OCstops=[]
    for i in range(len(OCperiods)):
        td=datetime.datetime(int(OCperiods[i].split('-')[0]),1,1)-datetime.datetime(1961,1,1)
        OCstarts.append(td.days)
        td=datetime.datetime(int(OCperiods[i].split('-')[1])+1,1,1)-datetime.datetime(1961,1,1)
        OCstops.append(td.days)

    SP[2]=[]
    # build list of timesteps to write OC for
    for i in range(len(OCstops)):
        ts=range(timesteps+1)[OCstarts[i]:OCstops[i]]
        for step in ts:
            SP[2].append(step)
        
ofp=open('BEC_gsflow_shortened.oc','w')
ofp.write('#Black Earth Creek Simulation - GSFLOW\n')
ofp.write(header)
ofp.write('HEAD SAVE UNIT 30\n')
for i in SP.iterkeys():
    for timestep in SP[i]:
        ofp.write('PERIOD   %s STEP   %s\n' %(i,timestep))
        if i==1 and timestep==0:
            ofp.write('    PRINT BUDGET\n')
            ofp.write('    SAVE BUDGET\n')
            ofp.write('    PRINT HEAD\n')
            ofp.write('    SAVE HEAD\n')
        else:
            ofp.write('    PRINT BUDGET\n')
            ofp.write('    SAVE HEAD\n')
ofp.close()

# launch GSFLOW
os.system('gsflow_develop_21Jun12.exe control\\%s' %(file2run))

# Make MOD2SMP input and run MOD2SMP
ofp=open('mod2smp.in','w')
ofp.write('ex.spc\n')
ofp.write('BEC_monitoring_well_coordinates_SMW.crd\n')
ofp.write('BEC_monitoring_well_coordinates_SMW.crd\n')
ofp.write('BEC14a_gsflow.hds\n')
ofp.write('f\n')
fname=file2run.split('.')
ofp.write('%s\n' %(timesteps+1))
ofp.write('8888\n')
ofp.write('d\n')
ofp.write('01/01/%s\n' %(startyr))
ofp.write('00:00:00\n')
ofp.write(file2run[:-7]+'ssf\n')
ofp.close()
os.system('mod2smp.exe <mod2smp.in')

# zipup PRMS results and .ggo files
zip=zipfile.ZipFile('BEC_run_%s.zip' %(jobnumber),'w',zipfile.ZIP_DEFLATED)
output=os.listdir('output')

for files in output:
    zip.write(os.path.join('output',files))

allfiles=os.listdir(os.getcwd())
ggofiles=[f for f in allfiles if f.lower().endswith('.ggo')]
ssffiles=[f for f in allfiles if f.lower().endswith('.ssf')]

for files in ggofiles:
    zip.write(files)
for files in ssffiles:
    zip.write(files)
zip.close()
