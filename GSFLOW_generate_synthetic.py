# Generates synthetic input data in front of real input data, to allow for model "spin-up"

import os
import calendar

datadir='input' # directory containing original .data input files
newdatadir='input_w_synthetic' # where to put new .data files

timepers=['1961-2000','2046-2065','2081-2100']
repeatpers=['1961-1961','2046-2046','2081-2081']
nyearss=[20,40,40]

for i in range(len(timepers)):
    timeper=timepers[i] # (str) 'yyyy-yyyy' time period to extend
    repeatper=repeatpers[i] # (str) 'yyyy-yyyy' time period repeat for synthetic data (01-01 to 12-31)
    nyears=nyearss[i] # (int) years of synthetic time to insert before beginning of real input dataset
    
    
    # get list of files
    allfiles=os.listdir(datadir)
    datafiles2edit=[f for f in allfiles if timeper in f and f.endswith('.data')]
    
    # setup time periods
    timeper=map(int,timeper.split('-'))
    newtimeper='%s-%s' %(timeper[0]-nyears,timeper[1])
    repeatper=map(int,repeatper.split('-'))
    cycles=int(float(nyears)/float(repeatper[1]-repeatper[0]+1))
    
    print 'Adding synthetic data for %s-%s...' %(timeper[0]-nyears,timeper[0])
    
    for files in datafiles2edit:
        print os.path.join(datadir,files),
        GCM,scenario,realization,timeper=files.split('.')[0:4]
        newfname='%s.%s.%s.%s.bec_ide.data' %(GCM,scenario,realization,newtimeper)
        
        data=open(os.path.join(datadir,files)).readlines()
        data2repeat=[]
        header=[]
        
        for line in data:
            # find start
            try:
                year=int(line.split()[0])
            except:
                header.append(line)
                continue
            if repeatper[0]<=year<=repeatper[1]:
                data2repeat.append(line)
                
        Leapdaywritten=False
        syntheticdata=[]
        for n in range(cycles):
            
            for line in data2repeat:
                splitline=line.split()
                month,day=map(int,splitline[1:3])
                newyear=map(int,newtimeper.split('-'))[0]+n
                Leapyear=calendar.isleap(newyear)
                if Leapyear:
                    if month==2 and day==29:
                        Leapdaywritten=True
                        newline='%s%s' %(newyear,line[4:])
                        syntheticdata.append(newline)
                    if month==3 and day==1:
                        if not Leapdaywritten: # copy data from March1 to Feb 29
                            newline='%s  %s  %s  %s' %(newyear,2,29,line[12:])
                            syntheticdata.append(newline)
                            newline=newline='%s%s' %(newyear,line[4:])
                            syntheticdata.append(newline)
                        else:
                            newline=newline='%s%s' %(newyear,line[4:])
                            syntheticdata.append(newline)
                            # reset flag until 2-29 activates again
                            Leapdaywritten=False
                    else:
                        newline=newline='%s%s' %(newyear,line[4:])
                        syntheticdata.append(newline)
                else:
                    newline='%s%s' %(newyear,line[4:])
                    syntheticdata.append(newline)
        
        print '-->\%s' %(os.path.join(newdatadir,newfname))
        ofp=open(os.path.join(newdatadir,newfname),'w')
        for line in header:
            ofp.write(line)
        for line in syntheticdata:
            ofp.write(line)
        for line in data:
            # find start
            try:
                year=int(line.split()[0])
            except:
                continue
            ofp.write(line)
        ofp.close()
        

    
            
            
    