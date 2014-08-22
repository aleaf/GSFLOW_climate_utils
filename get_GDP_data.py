'''
Retrieve GDP data (using submitFeatureWeightedGridStatistics method in pyGDP)
for polygons specified by a shapefile

option to download datasets in each URI individually or together
'individually' may be somewhat wasteful, because the Area Grid Statistics need to be recalculated for each dataset
(and this appears to be time consuming)
but it is more robust if the server is having connection issues
(i.e., if the server is disconnecting before the large file with all datasets can be downloaded)
and may be the only way to go for large datasets (for example, 880 hrus by 40 years for Fox-Wolf model in 20th century,
which resulted in ~100+ MB output files for each gcm-scenario dataset)
'''

import os
import pyGDP
import time
import shutil
import traceback

pyGDP = pyGDP.pyGDPwebProcessing()

TestRun = False # turn this on to retrive a limited dataset for testing

zipped_shapefiles = ['D:/ATLData/Fox-Wolf/hrus_final.zip'] # list of zipped shapefiles (one zip file per shapefile)
outpath = 'D:/ATLData/Fox-Wolf/GDP/' # outfiles and recfile will be saved here
recfile = 'D:/ATLData/Fox-Wolf/GDP/GDP_request_recfile.txt' # record of all files received
attribute = 'GRID_CODE' # attribute with unique identifier for each weather station
realization = '-01' # enter a string identifying the realization
parameters = ['prcp', 'tmin', 'tmax']

# GDP settings
URI_designator = 'wicci' # search for all URIs containing this text
URI_names = ['20c3m', 'sres_early', 'sres_late'] # get data for URIs ending in these names (using os.path.split)
retry_getDataType_after = 5 # seconds to wait before trying pyGDP.getDataType(URI) again
restart_submit_after = 10 # minutes to wait before restarting after failed submit
download_datasets = 'individually' # 'together' or 'individually' (see notes above)


def submit_request(shapefile, URI, d, timeStart, timeEnd, attribute, values, ofp, mode='individually'):
    '''
    calls pyGDP.submitFeatureWeightedGridStatistics()
    either with single dataset, or list of datasets (depending on mode)
    resubmits after waiting in the case of a server error
    '''

    if mode == 'together':
        outfile = os.path.join(outpath, os.path.split(URI)[1] + '.csv')
        record = ',all datasets written to {}\n'.format(outfile)
    else:
        outfile = os.path.join(outpath, d+'_'+URI[-5:]+'.csv')
        record = ',{}\n'.format(d)

    done = False

    # while loop to wait and then resubmit if server/network fails
    # for some reason this appears to work after an exception (the files keep coming in with the right names), but printing to the output screen stops
    while not done:
        try:
            cida_handle = pyGDP.submitFeatureWeightedGridStatistics(shapefile, URI, d, timeStart, timeEnd, attribute, values)
            done = True
        except Exception, e:
            print e
            print traceback.format_exc()
            print "waiting {} min before trying again...".format(restart_submit_after)
            time.sleep(restart_submit_after * 60) # wait n seconds before attempting to submit again
            print "resubmitting..."
        else:
            ofp.write(cida_handle + record)
            shutil.move(cida_handle, outfile)
            print "saved as: {}".format(outfile)


# make outpath if it doesn't exist
if not os.path.isdir(outpath):
    os.makedirs(outpath)

# restart
if os.path.isfile(recfile):
    recfile_info = open(recfile, 'r').readlines()

    restart = True
    print "\nRestart file {} found...".format(recfile)
    
    rec_URIs = []
    for line in recfile_info:
        if 'usgs.gov' in line:
            rec_URIs.append(line[:-1])
    try:
        last_URI = rec_URIs[-1]
        print "last URI: {}".format(last_URI)

        rec_datatypes = []
        for line in recfile_info[1:]:
            if 'usgs.gov' not in line and last_URI[-5:] in line:
                datatype = line.strip().split(',')[1]
                rec_datatypes.append(datatype)
        last_datatype = rec_datatypes[-1]
        print "last last_datatype: {}".format(last_datatype)
    except:
        if len(recfile_info) < 2: # if only the header has been written, start over
            print 'Restart file empty, starting from beginning...'
            restart = False
        else:
            print 'Problem reading restart file {}'.format(recfile)
            quit()

else:
    restart = False
    print "\nNo restart file found, starting from beginning..."

#upload all shapefiles in the shp folder if they don't exist

GDPshapefiles = pyGDP.getShapefiles()
for shp in zipped_shapefiles:
    try:
        pyGDP.uploadShapeFile(shp)
        print "\nuploaded {} to server ".format(shp)
    except:
        print "\nshapefile {} found on server".format(shp)
        continue

shapefiles=['upload:' + os.path.split(f)[1][:-4] for f in zipped_shapefiles]

shapefile = shapefiles[0] # for now just using one shapefile

# in the this case, the station identifiers are just consecutive integers
values = pyGDP.getValues(shapefile, attribute)
#values = map(str,sorted(map(int, values))) # could enforce order, but doesn't seem to make a difference

# Search for datasets
print "\nGetting datasets and datatypes..."
dataSetURIs = pyGDP.getDataSetURI(anyText=URI_designator)
datasets = dataSetURIs[1][2] # this probably needs to be hard-coded based on results of line above
# get datasets that contain the specified URI names
datasets = [[d for d in datasets if os.path.split(d)[1] == n][0] for n in URI_names]

if len(datasets) > 0:
    print '\nFound:'
    for n in datasets:
        print '{}'.format(n)

# in case of restart, trim already-processed entries from datasets
if restart:
    if download_datasets == 'individually':
        rec_URIs.pop() # if downloading together and last URI is in recfile, the dataset downloaded OK
    datasets=[d for d in datasets if d not in rec_URIs]

# keep a master record files of processed datasets
# if restarting, apend existing rec file like it was a continuous run
ofp = open(recfile, 'w')
if restart:
    for line in recfile_info:
        ofp.write(line)
else:
    ofp.write('CIDA_output_handle, datatype\n')

# Loop through the datasets (Time periods)
print '\nGetting DataTypes for each URI...'
for URI in datasets:

    # Get list of datatypes based on realization and parameters specified above
    datatypes_list = []
    
    # in case the server bombs out, try again
    dTypes = False
    while not dTypes:
        try:
            print '\n{}'.format(URI)
            dataTypes = pyGDP.getDataType(URI)
            dTypes = True

        except Exception, e:
            print e
            print "trying again in a moment..."
            time.sleep(5)
            print "asking again for dataTypes..."
    
    if len(dataTypes) == 0:
        print "Error! no datasets returned."

    for d in dataTypes:
        for p in parameters:
            if realization in d and p in d:
                datatypes_list.append(d)
                
    if restart:
        datatypes_list = [d for d in datatypes_list if d not in rec_datatypes]
        if len(datatypes_list) == 0:
            rec_datatypes = []
            continue

    # could add something here to print out list of datatypes after restart    
    
    # Get time periods to run for list of datatypes
    print '\nDataTypes and timeRanges (excluding those already downloaded):'
    timeranges=[]
    for d in datatypes_list:
        timeRange = pyGDP.getTimeRange(URI, d)
        timeranges.append(timeRange)
        print '{}\t-->\t{} - {}'.format(d, timeRange[0], timeRange[1])

    # assumes that time ranges are all the same for all datatypes
    # loop to get rid of stupid empties in timeranges list
    for n in timeranges:
        try:
            timeStart = n[0]
            timeEnd = n[1]
            break
        except IndexError:
            continue    
    if TestRun: 
        # kludgy hard code to test only first timestep for wicci
        if timeStart == '1961-01-01T00:00:00Z':
            timeEnd = '1961-01-02T00:00:00Z'
        if timeStart == '2046-01-01T00:00:00Z':
            timeEnd = '2046-01-02T00:00:00Z'     
        if timeStart == '2081-01-01T00:00:00Z':
            timeEnd = '2081-01-02T00:00:00Z'

    
    # Decide whether or not to write the URI to the recfile (in case of restart)
    # otherwise the recfile structure will be wrong, affecting future restarts
    if restart:
        if URI != last_URI:
            ofp.write(URI+'\n')
    else:
        ofp.write(URI+'\n')

    if download_datasets == 'together':
        print "\nsubmitting FeatureWeightedGridStatistics request with all DataTypes, from {} to {}..."\
            .format(timeStart, timeEnd)
        submit_request(shapefile, URI, datatypes_list, timeStart, timeEnd, attribute, values, ofp)

    elif download_datasets == 'individually':
        print "\nsubmitting FeatureWeightedGridStatistics requests for each DataType, from {} to {}..."\
            .format(timeStart, timeEnd)

        for d in datatypes_list:
            print '\n{}'.format(d)
            submit_request(shapefile, URI, d, timeStart, timeEnd, attribute, values, ofp)

ofp.close()
print "finished !"