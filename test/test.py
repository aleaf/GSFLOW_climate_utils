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

if __name__ == '__main__':
    test_statistics()