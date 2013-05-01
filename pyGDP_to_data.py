# Program to take WICCI csv files downloaded from the USGS-CIDA Geo Data Portal and convert to PRMS format
# Performs same task as John Walker's program wicci_to_cbh, but for GDP output generated by pyGDP, which is seperated into seperate csv files for each Time Period-GCM-Scenario-Variable combination (as opposed to the web GDP interface, which produces lumped files for each time period).

import os
import numpy as np
from collections import defaultdict

csvdir='outfiles' # directory containing downloaded files from wicci
datadir='input' # directory for converted files

print "Getting list of csv files..."
try:
    allfiles=os.listdir(csvdir)
except OSError:
    print "can't find directory with GDP files"
    

csvs=[f for f in allfiles if f.lower().endswith('.csv')]

# Build dictionary, 1 entry for each gcm-scenario-realization-time period combination
# each entry should then have a tmax, tmin, and prcp file
# some of these lines below are hard-coded and may need to be looked at if the input changes
combinations=defaultdict()
for cf in csvs:
    (scenario,gcm,par,realtime)=cf.split('-')
    realization=int(realtime[:2])
    if '20c3m' in realtime:
        timeper='1961-2000'
    elif 'early' in realtime:
        timeper='2046-2065'
    elif 'late' in realtime:
        timeper='2081-2100'
    else:
        raise Exception("Cannot parse time period from filename")
    
    combination='%s.%s.%s.%s.bec_ide' %(gcm,scenario,realization,timeper)
    # create dictionary entry for each scenario-gcm-realization-timeperiod combination
    try:
        combinations[combination].append(cf)
    except KeyError:
        combinations.setdefault(combination, []).append(cf)        

# make sure that all of the files are there
for c in combinations.iterkeys():
    if len(combinations[c])!=3:
        raise IndexError("missing an input file for " + c + "!")


print "Converting files to data format..."

filenum=0
# For each combination, combine parameter files into one output file
for combination in combinations.iterkeys():
    
    data=dict()
    num_attribs=dict()
    print combination + '-'*20
    
    # build dictionaries with entries for each parameter file in combination
    for file in combinations[combination]:
        print file
    
        csvpath=os.path.join(csvdir,file)
        f=open(csvpath,'r')
        name=f.readline()
        values=f.readline()
        parline=f.readline()
        t=False
        p=False    
        if "MEAN(mm)" in parline:
            par='prcp'
        elif "MEAN(C)" in parline:
            if 'tmin' in file:
                par='tmin'
            elif 'tmax' in file:
                par='tmax'
        else:
            raise Exception("Error: unrecognized parameter!")
        
        data[par]=np.genfromtxt(csvpath,dtype=None,skiprows=3,delimiter=',')
        num_attribs[par]=len(data[par][0])-1
    
    # open output file and write headers
    par_order=['tmax','tmin','prcp'] # enforce this order
    outpath=os.path.join(datadir,combination)
    ofp=open(outpath+'.data','w')
    ofp.write('created by pyGDP_to_data.py\n')
    for n in par_order:
        ofp.write(n+' '+str(num_attribs[n])+'\n')

    ofp.write('solrad 0\npan_evap 0\nrunoff 0\nform_data 0\n')
    ofp.write('#'*40+'\n')
    
    print('\n...\t'),
    
    # loop through lines in input files, converting and writing to output, line by line
    # each new line in output has date, then tmax, tmin, prcp
    
    for i in range(len(data[par])):
        
        # explode date from first parameter and add
        datetime=list(data[par][0])[0]
        (year,month,day)=datetime[:10].split('-')
        (h,m,s)=datetime[11:-1].split(':')
        newline=map(int,[year,month,day,h,m,s])        
        
        # add values for each parameter in par_order
        for par in par_order:
            
            line=list(data[par][i])
            for value in line[1:]:
                if par=='prcp':
                    if value<=5e-5:
                        valueIn="{0:.2f}".format(0)
                    else:
                        valueIn="{0:.2f}".format(value/25.4) # mm/in
                elif 't' in par:
                    valueIn="{0:.2f}".format(value*(9.0/5.0)+32.0) # C to F
                newline.append(valueIn)
                
        newline='  '.join(map(str,newline)) + '\n'
        ofp.write(newline)
    ofp.close()
    filenum+=1.0
    print str("{0:.0f}".format(100*filenum/len(combinations)+"% Done"
print "All files converted!"
    