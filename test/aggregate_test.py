
import sys
sys.path.append('../Postprocessing')
import os
import glob
import numpy as np
import pandas as pd
import zipfile as zf


def get_info(zipfile_handle):
    """gets the gcm, scenario, and period info for a zipfile

    Returns
    -------
    gcm : str
    scenario : str
    realization : str
    period : str
    """
    statvarname = [f for f in zipfile_handle.namelist() if 'statvar' in f]
    if len(statvarname) > 0:
        return os.path.split(statvarname[0])[1].split('.')[0:4]
    else:
        print("no statvar file in archive!")

def test_aggregated_ggo():
    """Test that the gage package results in an
    aggregated file match those in the original files"""

    resultspath = '/Users/aleaf/Documents/BlackEarth/run3_basecase/results'
    aggregated_resultspath = '/Users/aleaf/Documents/BlackEarth/run3_basecase/ggo'
    nfiles2test = 2

    for path in [aggregated_resultspath, resultspath]:
        assert os.path.isdir(path)

    zipfiles = [f for f in glob.glob(resultspath +'/*.zip') if os.path.getsize(f) > 0]
    zipfiles = [zf.ZipFile(f) for f in zipfiles]

    # get gcm and scenario for each zipfile
    zip_gcm_scen = {}
    for z in zipfiles:
        if os.path.getsize > 0:
            gcm, scen, realization, period = get_info(z)
            zip_gcm_scen['{}.{}'.format(gcm, scen)] = z.filename

    files2test = glob.glob('{}/*.csv'.format(aggregated_resultspath))[0:nfiles2test]
    for aggregated_file in files2test:
        # get var and scenario from aggregated file name; read it in
        var, scenario = os.path.split(aggregated_file)[1].split('.')[0:2]
        agg = pd.read_csv(os.path.join(aggregated_resultspath, aggregated_file))

        gcms = agg.columns[1:]
        col = 2 # flow is in column 2

        for gcm in gcms:
            # get the zipfile with original information for aggregated file
            # get the original file out of the zipfile
            try:
                z = zf.ZipFile(zip_gcm_scen['{}.{}'.format(gcm, scenario)])
            except KeyError:
                continue
            f = [f for f in z.namelist() if var in f]
            if len(f) > 0:
                ff = z.open(f[0])
                orig = pd.read_csv(ff, skiprows=2, delim_whitespace=True, header=None)

            # verify that the column in the original file matches the column in the aggregated file
            if not np.all(np.isnan(agg[gcm])):
                assert (orig[2] - agg[gcm]).sum() < 1e-8

if __name__ == '__main__':
    test_aggregated_ggo()