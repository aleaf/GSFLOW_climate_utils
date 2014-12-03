import os
import climate_plots as cp
from Figures import ReportFigures
import pandas as pd

figures = ReportFigures()
figures.set_style(style='timeseries', width='single')

inputfolder = 'D:/ATLData/Fox-Wolf/data'
outpdf = 'D:/ATLData/Fox-Wolf/Fox-Wolf_growing_season.pdf'

csvs = [os.path.join(inputfolder, c) for c in os.listdir(inputfolder) if 'growing_season' in c]

dfs = {}
for c in csvs:
    df = pd.read_csv(c, index_col=0, parse_dates=True)
    scenario = os.path.split(c)[-1].split('_')[0]
    dfs[scenario] = df

props = {'sresa1b': {'color': 'Tomato', 'zorder': -2, 'alpha': 0.5},
        'sresa2': {'color': 'SteelBlue', 'zorder': -3, 'alpha': 0.5},
        'sresb1': {'color': 'Yellow', 'zorder': -1, 'alpha': 0.5},
        }


fig, ax = cp.timeseries(dfs, ylabel='Growing season length (days)', props=props, title='',
                        plotstyle=figures.plotstyle) # plotstyle dict as argument to override some Seaborn settings


fig.savefig(outpdf, dpi=300)