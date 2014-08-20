
import os
import pyGDP
import time

pyGDP=pyGDP.pyGDPwebProcessing()

TestRun = True # turn this on to retrive a limited dataset for testing

zipped_shapefiles = ['D:/ATLData/Fox-Wolf/hrus_final.zip'] # list of zipped shapefiles (one zip file per shapefile)
outpath = os.path.join('D:/ATLData/Fox-Wolf/GDP/') # outfiles and recfile will be saved here
recfile = 'D:/ATLData/Fox-Wolf/GDP/GDP_request_recfile.txt' # record of all files received
attribute = 'GRID_CODE' # attribute with unique identifier for each weather station
realization ='-01' # enter a string identifying the realization
parameters = ['prcp', 'tmin', 'tmax']


def restart_switch(outpath, recfile):
    # Need to figure out how to recover from intentional crash
    try:
        recfile_info=open(outpath+recfile,'r').readlines()
        return recfile_info
                        
    except IOError:
        return []
    
recfile_info = restart_switch(outpath, recfile)

if len(recfile)==0:
    print "Recfile empty or no existing rec file found, starting from beginning..."
    
if len(recfile_info)>0:
    restart = True
    print "restarting run from last file..."
    print "last URI and last datatype are:"    
    
    rec_URIs=[]
    for line in recfile_info:
        if 'usgs.gov' in line:
            rec_URIs.append(line[:-1])
    last_URI = rec_URIs[-1]
    print last_URI
    
    rec_datatypes=[]
    for line in recfile_info[1:]:
        if 'usgs.gov' not in line and last_URI[-5:] in line:
            rec_datatypes.append(line[:-11].split(',')[1])
    last_datatype=rec_datatypes[-1]
    print last_datatype
    
    
#upload all shapefiles in the shp folder if they don't exist

GDPshapefiles = pyGDP.getShapefiles()
for shp in zipped_shapefiles:
    try:
        print "uploading " + shp
        pyGDP.uploadShapeFile(shp)
    except:
        print "shapefile already exists on server"
        continue

shapefiles=['upload:'+f[:-4] for f in zipped_shapefiles]

shapefile=shapefiles[0] # for now just using one shapefile

# in the this case, the station identifiers are just consecutive integers
values = pyGDP.getValues(shapefile, attribute)
#values = map(str,sorted(map(int, values))) # could enforce order, but doesn't seem to make a difference

# Search for datasets
print "Getting datasets and datatypes..."
dataSetURIs = pyGDP.getDataSetURI(anyText='wicci')
datasets = dataSetURIs[1][2][0:3] # this probably needs to be hard-coded based on results of line above

# in case of restart, trim already-processed entries from datasets
if restart:
    rec_URIs.pop()
    datasets=[d for d in datasets if d not in rec_URIs]

# keep a master record files of processed datasets
# if restarting, apend existing rec file like it was a continuous run
ofp=open(outpath+recfile,'w')
if restart:
    for line in recfile_info:
        ofp.write(line)
else:
    ofp.write('CIDA_output_handle,datatype\n')

# Loop through the datasets (Time periods)   
for URI in datasets:
    print URI
    # Get list of datatypes based on realization and parameters specified above
    datatypes_list=[]
    
    # in case the server bombs out, try again
    dTypes=False
    while not dTypes:
        try:
            dataTypes = pyGDP.getDataType(URI)
            dTypes=True
        except Exception, e:
            print e
            print "trying again in a moment..."
            time.sleep(5)
            print "asking again for dataTypes..."
    
    if len(dataTypes)==0:
        print "Error! no datasets returned."
        
    for d in dataTypes:
        for p in parameters:
            if realization in d and p in d:
                print d
                datatypes_list.append(d)
                
    if restart:
        datatypes_list=[d for d in datatypes_list if d not in rec_datatypes]
        if len(datatypes_list)==0:
            rec_datatypes=[]
            continue
    # could add something here to print out list of datatypes after restart    
    
    # Get time periods to run for list of datatypes
    timeranges=[]
    for d in datatypes_list:
        timeRange = pyGDP.getTimeRange(URI, d)
        timeranges.append(timeRange)

    # assumes that time ranges are all the same for all datatypes
    # loop to get rid of stupid empties in timeranges list
    for n in timeranges:
        try:
            timeStart=n[0] 
            timeEnd=n[1]
            break
        except IndexError:
            continue    
    if TestRun: 
        # kludgy hard code to test only first timestep for wicci
        if timeStart=='1961-01-01T00:00:00Z':
            timeEnd = '1961-01-02T00:00:00Z'
        if timeStart=='2046-01-01T00:00:00Z':
            timeEnd = '2046-01-02T00:00:00Z'     
        if timeStart=='2081-01-01T00:00:00Z':
            timeEnd = '2081-01-02T00:00:00Z'   
        datatypes_list=datatypes_list[0:2]
    print "Start time is " + timeStart
    print "End time is " + timeEnd
    
    # Decide whether or not to write the URI to the recfile (in case of restart)
    # otherwise the recfile structure will be wrong, affecting future restarts
    if restart:
        if URI!=last_URI:
            ofp.write(URI+'\n')
    else:
        ofp.write(URI+'\n')
        
    print "submitting requests..."
    
    for d in datatypes_list:
        done=False
        # while loop to wait and then resubmit if server/network fails
        # for some reason this appears to work after an exception (the files keep coming in with the right names), but printing to the output screen stops
        while not done:
            try:
                outfile= pyGDP.submitFeatureWeightedGridStatistics(shapefile, URI, d, timeStart, timeEnd, attribute, values)
                done=True
            except Exception, e:
                print e
                print "waiting 10 min before trying again..."
                time.sleep(600) # wait n seconds before attempting to submit again
                print "resubmitting..."
            else:
                new_ofname=d+'_'+URI[-5:]+'.csv'
                ofp.write(outfile+','+new_ofname+'\n')
                os.rename(outfile,outpath+new_ofname)
                print d +URI[-5:]+ " finished"             
ofp.close()
print "finished !"