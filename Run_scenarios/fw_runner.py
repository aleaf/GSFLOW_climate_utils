'''
Example program to batch PRMS future climate runs without parallelization
'''

import os
from subprocess import Popen, PIPE

# make a list of PRMS control files for each run; assign to consecutive numbers starting at 0
controldir = 'D:/ATLData/Fox-Wolf/control'
controlfiles = sorted([os.path.join(controldir, f) for f in os.listdir(controldir) if f.lower().endswith('.control')])
executable = 'D:/ATLData/Fox-Wolf/prms_ws_9_17_2012.exe'
output_logfile = 'D:/ATLData/Fox-Wolf/PRMS_runs_output.txt'
outputdir = os.path.join(os.path.split(controldir)[0], 'output')

# make output dir if one doesn't exist
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

ofp = open(output_logfile, 'w')
# run PRMS
for c in controlfiles:
    print '{}\t{}'.format(executable, c)
    ofp.write('{}\t{}'.format(executable, c))

    p = Popen([str(executable), str(c)], stdout=PIPE, stderr=PIPE, stdin=PIPE)

    ofp.write(p.stdout.read())

ofp.close()
