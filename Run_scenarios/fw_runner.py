'''
Example program to run an instance of a PRMS model using HTCondor
this file should be in the main run folder (at the same level as the output, control, params, etc. folders)
in Condor batch file, call this file with an argument that is tied to the job number
i.e. in worker.bat, python runner.py %1; then in SUB file, arguments = $(Process)
'''

import os
import sys
import zipfile

# make a list of PRMS control files for each run; assign to consecutive numbers starting at 0
allfiles = os.listdir('control')
controlfiles = [f for f in allfiles if f.lower().endswith('.control')]
controlfiles = sorted(controlfiles)

# make output dir if one doesn't exist
if not os.path.isdir('output'):
    os.makedirs('output')

# get unique job number from Condor
jobnumber = int(sys.argv[1])

# assign control file based on job number
file2run=controlfiles[jobnumber].strip()
print "\nsetting up to run %s" %(file2run)

# launch PRMS
os.system('prms_ws_9_17_2012.exe control\\%s' %(file2run))

# zipup PRMS results and .ggo files
zip = zipfile.ZipFile('FW_run_%s.zip' %(jobnumber), 'w', zipfile.ZIP_DEFLATED)

for files in os.listdir('output'):
    zip.write(os.path.join('output', files))

zip.close()
