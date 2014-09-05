__author__ = 'aleaf'
'''
quick way to check PRMS output after single run
'''
import os
from PRMSio import statvarFile
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

statvarfile = 'D:/ATLData/Fox-Wolf/output/cccma_cgcm3_1.20c3m.1.1961-2000.fw.statvar.dat'

# instantiate object for statvarfile
sv = statvarFile(statvarfile)

# read statvarfile into pandas dataframe
df = sv.read2df()

# resample to annnual means
dfa = df.resample('A', how='mean')

pdf = PdfPages(statvarfile[:-4] + '.pdf')

# make a plot of each column (variable)
for c in dfa.columns:
    ax = dfa[c].plot(title=c)
    ax.set_xlabel('')
    pdf.savefig()
    plt.close()
pdf.close()