import sys
sys.path.append('../Postprocessing')
import os
import shutil
import numpy as np
import pandas as pd
import climate_stats as cs

def test_statistics():
    ## Test the summary statisctics method
    # Create the test cases
    if os.path.isdir('test'):
        shutil.rmtree('test')
    os.mkdir('test')
    testrange = pd.date_range('01-01-2014','12-31-2016')
    for i in range(3):
        df = pd.DataFrame({'{}'.format(i): np.arange(len(testrange), dtype=float)+i}, index=testrange)
        df.index.name = 'Date'
        df.to_csv('test/test.{}.csv'.format(i))

    compare_periods = np.array([[2015, 2015],[2016, 2016]])
    baseline_period = [2014, 2014]
    stat = 'mean_monthly'
    boxwidth=0.4
    xtick_freq=1
    csvs={'0': 'test/test.0.csv',
          '1': 'test/test.1.csv',
          '2': 'test/test.2.csv'}

    ## test monthly sums
    boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'mean_monthly', baseline_period,
                                                   calc='sum', quantile=None)

    # test monthly baseline
    assert baseline[0] == np.sum(np.arange(1, 32))
    assert baseline[-1] == np.sum(np.arange(365-30, 366))
    # test monthly boxcolumns
    assert boxcolumns['January 2015-2015'].mean() == np.arange(366, 366+31).sum()
    assert boxcolumns['January 2016-2016'].mean() == np.arange(731, 731+31).sum()

    ## test annual sums
    boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'mean_annual', baseline_period,
                                                   calc='sum', quantile=None)
    # test annual baseline
    assert baseline[0] == np.arange(1, 366).sum()
    # test annual boxcolumns
    assert boxcolumns[0].mean() == np.arange(366, 366+365).sum()
    assert boxcolumns[1].mean() == np.arange(366+365, 366+731).sum() # 2016 is a leap year!

    ## test monthly means
    boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'mean_monthly', baseline_period,
                                                   calc='mean', quantile=None)

    # test monthly baseline
    assert baseline[0] == np.mean(np.arange(1, 32))
    assert baseline[-1] == np.mean(np.arange(365-30, 366))
    # test monthly boxcolumns
    assert boxcolumns['January 2015-2015'].mean() == np.arange(366, 366+31).mean()
    assert boxcolumns['January 2016-2016'].mean() == np.arange(731, 731+31).mean()

    ## test annual means
    boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'mean_annual', baseline_period,
                                                   calc='mean', quantile=None)
    # test annual baseline
    assert baseline[0] == np.arange(1, 366).mean()
    # test annual boxcolumns
    assert boxcolumns[0].mean() == np.arange(366, 366+365).mean()
    assert boxcolumns[1].mean() == np.arange(366+365, 366+731).mean() # 2016 is a leap year!

    ## test annual sums timeseries
    dfs = cs.annual_timeseries(csvs, ['0', '1', '2'], 1, 'mean_annual', calc='sum')

    # test annual boxcolumns
    assert dfs['1'].ix['2015-01-01', '1'] == np.arange(366, 366+365).sum()
    assert dfs['2'].ix['2016-01-01', '2'] == (np.arange(366+365, 366+731) + 1).sum() # 2016 is a leap year!

    ## test annual means timeseries
    dfs = cs.annual_timeseries(csvs, ['0', '1', '2'], 1, 'mean_annual', calc='mean')

    # test annual boxcolumns
    assert dfs['1'].ix['2015-01-01', '1'] == np.arange(366, 366+365).mean()
    assert dfs['2'].ix['2016-01-01', '2'] == (np.arange(366+365, 366+731) + 1).mean() # 2016 is a leap year!

    ## test annual quantiles timeseries
    dfs = cs.annual_timeseries(csvs, ['0', '1', '2'], 1, 'quantile', quantile=0.1)
    assert dfs['1'].ix['2015-01-01', '1'] == np.percentile(np.arange(366, 366+365), 10)

    ##test annual quantiles box plots
    boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'quantile', baseline_period,
                                           quantile=0.1)
    assert boxcolumns[0].mean() - np.percentile(np.arange(366, 366+365), 10) < 1e-8

    ## test normalize option
    for calc in ['mean', 'sum']:
        boxcolumns, baseline = cs.period_stats(csvs, compare_periods, 'mean_annual', baseline_period,
                                                   calc=calc, quantile=None, normalize_to_baseline=True)
        assert boxcolumns[2, 0] == 200.0


    # test code for time series means on real data
    csvs={'sresa1b': 'Brew6_cfs.sresa1b.csv',
          'sresa2': 'Brew6_cfs.sresa2.csv',
          'sresb1': 'Brew6_cfs.sresb1.csv'}

    gcms = ['cccma_cgcm3_1',
              'cccma_cgcm3_1_t63',
              'cnrm_cm3',
              'csiro_mk3_0',
              'csiro_mk3_5',
              'gfdl_cm2_0',
              'giss_aom',
              'giss_model_e_r',
              'iap_fgoals1_0_g',
              'miroc3_2_hires',
              'miub_echo_g',
              'mpi_echam5',
              'mri_cgcm2_3_2a']

    dfs = cs.annual_timeseries(csvs, gcms, 20, 'mean_annual', calc='mean')

    pn = pd.Panel(dfs)
    df = pn.to_frame(filter_observations=False)
    stacked = df.stack().reset_index()
    stacked['year'] = [ts.year for ts in pd.to_datetime(stacked.major)]
    dfm = df.groupby(level=0).mean()

    assert (dfm['sresa1b'] - dfs['sresa1b'].mean(axis=1)).sum() < 1e-8


if __name__ == '__main__':
    test_statistics()