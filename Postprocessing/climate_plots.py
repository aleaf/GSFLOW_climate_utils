__author__ = 'aleaf'

import sys
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patheffects as PathEffects
import seaborn as sb
import textwrap
import calendar
import climate_stats as cs
import GSFLOW_utils as GSFu


#--modify the base rcParams for a few items
newparams = {'font.family': 'Univers 57 Condensed Light',
             'legend.fontsize': 8,
             'axes.labelsize': 9,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8,
             'pdf.fonttype': 42,
             'pdf.compression': 0,
             'axes.formatter.limits': [-7, 9]}

# Update the global rcParams dictionary with the new parameter choices
plt.rcParams.update(newparams)

# set/modify global Seaborn defaults
# update any overlapping parameters in the seaborn 'paper' style with the custom values above
sb.set_context("paper", rc=newparams)




class ReportFigures():

    # figure sizes (based on 6 picas per inch, see USGS Illustration Standards Guide, p 34
    default_aspect = 6 / 8.0 # h/w
    tall_aspect = 7 / 8.0
    singlecolumn_width = 21/6.0
    doublecolumn_width = 42/6.0

    # month abbreviations
    month = {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.', 5: 'May', 6: 'June',
             7: 'July', 8: 'Aug.', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.'}


    def __init__(self, mode, compare_periods, baseline_period, gcms, spinup,
                 aggregated_results_folder, output_folder,
                 var_name_file,
                 timeseries_properties,
                 variables_table=pd.DataFrame(),
                 synthetic_timepers=None,
                 exclude=None,
                 default_font='Univers 57 Condensed',
                 title_font='Univers 67 Condensed',
                 title_size=9,
                 legend_font='Univers 67 Condensed',
                 legend_titlesize=8,
                 plotstyle={},
                 box_colors=['SteelBlue', 'Khaki']):

        '''
        ``plotstyle``: Dictionary containing rc settings for overriding Seaborn defaults
            (<see http://web.stanford.edu/~mwaskom/software/seaborn/tutorial/aesthetics.html>)
        '''

        plots_folder = os.path.join(output_folder, mode)
        if not os.path.isdir(plots_folder):
            os.makedirs(plots_folder)

        print "\ngetting info on {} variables...".format(mode)
        varlist = GSFu.getvars(aggregated_results_folder, mode)
        varlist = [v for v in varlist if v not in exclude]

        self.mode = mode
        self.varlist = varlist
        self.gcms = gcms
        self.spinup = spinup
        try:
            self.var_info = GSFu.get_var_info(var_name_file)
        except:
            self.var_info = {}
        self.aggregated_results_folder = aggregated_results_folder
        self.output_base_folder = output_folder
        self.output_folder = os.path.join(output_folder, mode)
        self.mode = mode
        self.compare_periods = compare_periods
        self.baseline_period = baseline_period
        self.synthetic_timepers = synthetic_timepers
        self.timeseries_properties = timeseries_properties
        self.variables_table = variables_table
        self.dates = ['-'.join(map(str, per)) for per in compare_periods]
        self.box_colors = box_colors

        # font formatting (e.g. for USGS reports)
        self.default_font = default_font # for setting seaborn styles for each plot
        self.title_font = title_font # font for plot titles
        self.title_size = title_size
        self.legend_font = legend_font
        self.legend_titlesize = legend_titlesize
        self.legend_fontsize = self.legend_titlesize - 1

        # default plot settings
        self.plotstyle = {'font.family': 'Univers 57 Condensed',
                          'axes.linewidth': 0.5,
                          'axes.labelsize': 8,
                          'axes.titlesize': 9,
                          "grid.linewidth": 0.5,
                          'xtick.major.width': 0.5,
                          'ytick.major.width': 0.5,
                          'xtick.minor.width': 0.5,
                          'ytick.minor.width': 0.5,
                          'xtick.labelsize': 8,
                          'ytick.labelsize': 8,
                          'xtick.direction': 'in',
                          'ytick.direction': 'in',
                          'xtick.major.pad': 3,
                          'ytick.major.pad': 3,
                          'axes.edgecolor' : 'k',
                          'figure.figsize': (self.doublecolumn_width, self.doublecolumn_width * self.default_aspect)}

        # update default plot settings with any specified
        self.plotstyle.update(plotstyle)


        if not self.variables_table.empty:

            # if a variables table (DataFrame) was given, convert any empty fields to strings
            self.variables_table = self.variables_table.fillna(value='')



    def name_output(self, var, stat, type, quantile=None):
        # scheme to organize plot names and folders
        if quantile:
            stat = 'Q{:.0f}0'.format(10*(1-quantile))

        output_folder = os.path.join(self.output_folder, stat, type)
        os.makedirs(output_folder) if not os.path.isdir(output_folder) else None

        outfile = os.path.join(output_folder, '{}_{}_{}.pdf'.format(var, stat, type))
        return outfile


    def plot_info(self, var, stat, plottype, quantile=None):
        # Set plot titles and ylabels
        '''
        # set on the fly by reading the PRMS/GSFLOW variables file
        if self.variables_table.empty or self.mode != 'statvar' and self.mode != 'csv':
            title, xlabel, ylabel, calc = GSFu.set_plot_titles(var, self.mode, stat, self.var_info,
                                                         self.aggregated_results_folder,
                                                         plottype='box', quantile=quantile)
        # set using DataFrame from pre-made table
        else:
        '''
        info = self.variables_table[(self.variables_table['variable'] == var)
                                    & (self.variables_table['stat'] == stat)
                                    & (self.variables_table['plot_type'] == plottype)]
        try:
            title, xlabel = info.iloc[0]['title'], info.iloc[0]['xlabel']
            ylabel = '{}, {}'.format(info.iloc[0]['ylabel_0'], info.iloc[0]['ylabel_1'])
            calc = info.iloc[0]['calc']

        except:
            title, xlabel, ylabel, calc = GSFu.set_plot_titles(var, self.mode, stat, self.var_info,
                                                         self.aggregated_results_folder,
                                                         plottype='box', quantile=quantile)
        return title, xlabel, ylabel, calc


    def make_box(self, csvs, var, stat, quantile=None):

        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'box', quantile=quantile)

        # calculate montly means for box plot
        boxcolumns, baseline = cs.period_stats(csvs, self.compare_periods, stat, self.baseline_period,
                                               calc=calc, quantile=quantile)



        # make box plot
        if 'month' in stat:

            # settings to customize Seaborn "ticks" style (i.e. turn off grid)
            rcparams = {'figure.figsize': (self.doublecolumn_width, self.doublecolumn_width * self.tall_aspect),
                        'lines.linewidth': 1.0}

            fig, ax = sb_box_monthly(boxcolumns, baseline, self.compare_periods, ylabel,
                                     color=self.box_colors, plotstyle=self.plotstyle,
                                     rcparams=rcparams,
                                     fliersize=plt.rcParams['figure.figsize'][0], linewidth=self.plotstyle['axes.linewidth'])

        else:
            # settings to customize Seaborn "ticks" style (i.e. turn off grid)
            rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.tall_aspect),
                        'lines.linewidth': 0.5}

            fig, ax = sb_box_annual(boxcolumns, baseline, self.compare_periods, ylabel,
                                    color=self.box_colors, plotstyle=self.plotstyle,
                                    rcparams=rcparams,
                                    fliersize=plt.rcParams['figure.figsize'][0] * 1.5, linewidth=self.plotstyle['axes.linewidth'])

        self.figure_title(ax, title)
        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'box', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()


    def make_timeseries(self, csvs, var, stat, quantile=None):

        # Set plot titles and ylabels
        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'timeseries', quantile=quantile)

        # calculate annual means
        dfs = cs.annual_timeseries(csvs, self.gcms, self.spinup, stat, calc=calc, quantile=quantile)

        # settings to customize Seaborn "ticks" style (i.e. turn off grid)
        rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.default_aspect),
                    'axes.grid': False,
                    'grid.linestyle': ''}

        # make 'fill_between' timeseries plot with  mean and min/max for each year
        fig, ax = timeseries(dfs, ylabel=ylabel, props=self.timeseries_properties, Synthetic_timepers=self.synthetic_timepers,
                             plotstyle=self.plotstyle, rcparams=rcparams)

        self.figure_title(ax, title)

        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'timeseries', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()


    def make_violin(self, csvs, var, stat, quantile=None):
        # Set plot titles and ylabels
        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'box', quantile=quantile)

        # calcualte period statistics for violins
        boxcolumns, baseline = cs.period_stats(csvs, self.compare_periods, stat, self.baseline_period,
                                               calc=calc, quantile=quantile)

        # settings to customize Seaborn "ticks" style (i.e. turn off grid)
        rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.tall_aspect),
                    'lines.linewidth': 0.5,
                    'lines.markersize': self.singlecolumn_width}

        # make violin plot
        fig, ax = sb_violin_annual(boxcolumns, baseline, self.compare_periods, ylabel,
                                   color=self.box_colors, plotstyle=self.plotstyle,
                                   rcparams=rcparams)

        self.figure_title(ax, title)
        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'violin', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()


    def make_violin_legend(self):

        fig = plt.figure()

        plt.rcParams.update({'font.family': self.legend_font,
                             'legend.fontsize': self.legend_fontsize})

        ax = fig.add_subplot(111)
        handles=[]
        labels=[]
        for d in range(len(self.dates)):
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=self.box_colors[d]))
            labels.append(self.dates[d])
        handles.append(plt.Line2D(range(10), range(10), color='r', linewidth=2))
        labels.append('Baseline conditions ({}-{})'.format(self.baseline_period[0], self.baseline_period[1]))
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=1, linestyle=':'))
        labels.append('Upper/lower quartiles')
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=1, linestyle='--'))
        labels.append('Median')
        handles.append(plt.scatter(1, 1, c='k', s=12, marker='o'))
        labels.append('Values')

        figlegend = plt.figure(figsize=(3, 2))
        lg = figlegend.legend(handles, labels, title='EXPLANATION', loc='center')

        # update legend title size and font size to conform
        plt.setp(lg.get_title(), fontsize=self.legend_titlesize)

        outfile = os.path.join(self.output_base_folder, 'violin_legend.pdf')
        figlegend.savefig(outfile, dpi=300)
        plt.close('all')


    def make_box_legend(self):

        fig = plt.figure()

        plt.rcParams.update({'font.family': self.legend_font,
                             'legend.fontsize': self.legend_fontsize})

        handles=[]
        labels=[]
        for d in range(len(self.dates)):
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=self.box_colors[d]))
            labels.append(self.dates[d])
        handles.append(plt.Line2D(range(10), range(10), color='r', linewidth=2))
        labels.append('Baseline conditions ({}-{})'.format(self.baseline_period[0], self.baseline_period[1]))
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=1))
        labels.append('Boxes represent quartiles;'.format(self.baseline_period[0], self.baseline_period[1]))
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=1))
        labels.append('Whiskers represent 1.5x the interquartile range')
        handles.append(plt.scatter(1, 1, c='k', s=6, marker='D'))
        labels.append('Outliers')

        figlegend = plt.figure(figsize=(3, 2))
        lg = figlegend.legend(handles, labels, title='EXPLANATION', loc='center')

        # update legend title size and font size to conform
        plt.setp(lg.get_title(), fontsize=self.legend_titlesize)


        outfile = os.path.join(self.output_base_folder, 'box_legend.pdf')
        figlegend.savefig(outfile, dpi=300)
        plt.close('all')


    def make_timeseries_legend(self):

        plt.close('all')
        plt.rcParams.update({'font.family': self.legend_font,
                             'font.size': self.legend_fontsize,
                             'axes.linewidth': 0.5,
                             'axes.edgecolor': 'k'})

        #                     'xtick.major.size': 0,
        #                     'ytick.major.size': 0,
        #                     'grid.linestyle': ''})

        fig = plt.figure(1, (self.singlecolumn_width, self.singlecolumn_width * self.default_aspect * 0.7))

        grid = ImageGrid(fig, #111, # similar to subplot(111)

                         # set the rectangle for the grid relative to the canvas size (x=1, y=1)
                         # left, bottom, width, height
                         rect=[0.05, 0.02, 0.3, 0.6],
                         nrows_ncols = (1, 3), # creates 2x2 grid of axes
                         axes_pad=0.1) # pad between axes in inch

        scen = self.timeseries_properties.keys()
        for i in range(grid.ngrids):

            # background color for min/max
            grid[i].axvspan(xmin=0, xmax=1, facecolor=self.timeseries_properties[scen[i]]['color'],
                            alpha=self.timeseries_properties[scen[i]]['alpha'],
                            linewidth=0, zorder=0)

            # lines to represent means
            l = grid[i].axhline(y=0.5, xmin=0, xmax=1, color=self.timeseries_properties[scen[i]]['color'],
                                linewidth=4, zorder=1)
            l.set_path_effects([PathEffects.withStroke(linewidth=4, foreground="k")])

            # remove the ticks and size the plots
            grid[i].set_xticks([])
            grid[i].set_yticks([])
            grid[i].set_xlim(0, 0.5)

            # scenario labels
            title = scen[i].replace('sres', '').capitalize()
            grid[i].set_title(title, loc='left', fontsize=self.legend_fontsize, family=self.legend_font)

        # fade area to denote model spin-up
        grid[0].axvspan(xmin=0, xmax=0.20, facecolor='1.0',
                            alpha=0.8,
                            linewidth=0, zorder=2)
        grid[0].annotate('Screening indicates periods of synthetic data used for model spin-up',
                         xy=(0.18, 0), xycoords='data', xytext=(0.5, -0.35), ha='left', va='center',
                         arrowprops=dict(arrowstyle='-', linewidth=0.5, relpos=(0, .4)))


        # Labels for max/mean/min
        grid[i].text(1.2, 1, 'Maximum', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)
        grid[i].text(1.2, .5, 'Mean from General Circulation Models', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)
        grid[i].text(1.2, 0, 'Minimum', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)

        grid[2].text(0, 1.8, "EXPLANATION", ha='right', fontsize=self.legend_titlesize, family=self.legend_font)
        grid[0].text(0, 1.4, "Emissions Scenarios", ha='left', family=self.legend_font, fontsize=self.legend_fontsize)
        plt.title('stuff')
        '''
        #fig.subplots_adjust(top=0.5, left=-.25, bottom=0.1)
        with sb.axes_style("white"):
            # left, bottom, width, height
            fig2 = plt.figure(figsize=(6, 6))
            bbox = fig2.add_axes([0., -1., 8, 8], frameon=False, xticks=[],yticks=[])
        '''

        outfile = os.path.join(self.output_base_folder, 'timeseries_legend.pdf')
        fig.savefig(outfile, dpi=300)
        plt.close()


    def figure_title(self, ax, title, zorder=200):
        wrap = 60
        title = "\n".join(textwrap.wrap(title, wrap)) #wrap title
        '''
        ax.text(.025, 1.025, title,
                horizontalalignment='left',
                verticalalignment='bottom',
                transform=ax.transAxes, zorder=zorder)
        '''
        # with Univers 47 Condensed as the font.family, changing the weight to 'bold' doesn't work
        # manually specify a different family for the title
        ax.set_title(title.capitalize(), family='Univers 67 Condensed', zorder=zorder, loc='left')


#############
# Functions
##############


def thousands_sep(ax):
    '''
    format the ticknumbers
    '''

    # so clunky, but this appears to be the only way to do it
    if -10 > ax.get_ylim()[1] or ax.get_ylim()[1] > 10:
        fmt = '{:,.0f}'
    elif -10 < ax.get_ylim()[1] < -1 or 1 > ax.get_ylim()[1] > 10:
        fmt = '{:,.1f}'
    elif -1 < ax.get_ylim()[1] < -.1 or .1 > ax.get_ylim()[1] > 1:
        fmt = '{:,.1f}'
    else:
        fmt = '{:,.2e}'

    def format_axis(y, pos):
        y = fmt.format(y)
        return y

    ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))


def ignore_outliers_in_yscale(dfs, factor=2):
    # rescales plot to next highest point if the highest point exceeds the next highest by specified factor
    largest = np.array([])
    for df in dfs.itervalues():
        top_two = np.sort(df.fillna(0).values.flatten())[-2:]
        largest = np.append(largest, top_two)
    largest = np.sort(largest)
    if largest[-1] > factor * largest[-2]:
        l = largest[-2]
        # take log of max value, then floor to get magnitude
        # divide max value by magnitude and then multiply mag. by ceiling
        newlimit = np.ceil(l/10**np.floor(np.log10(l)))*10**np.floor(np.log10(l))
    else:
        newlimit = None
    return newlimit





def make_title(ax, title, zorder=1000):
    wrap = 60
    title = "\n".join(textwrap.wrap(title, wrap)) #wrap title
    '''
    ax.text(.025, 1.025, title,
            horizontalalignment='left',
            verticalalignment='bottom',
            transform=ax.transAxes, zorder=zorder)
    '''
    # with Univers 47 Condensed as the font.family, changing the weight to 'bold' doesn't work
    # manually specify a different family for the title
    ax.set_title(title.capitalize(), family='Univers 67 Condensed', zorder=zorder)


def timeseries(dfs, ylabel='', props=None, Synthetic_timepers=[],
               clip_outliers=True, xlabel='', title='',
               plotstyle={}, rcparams={}):

    # dfs = dict of dataframes to plot (one dataframe per climate scenario)
    # window= width of moving avg window in timeunits
    # function= moving avg. fn to use (see Pandas doc)
    # title= plot title, ylabel= y-axis label
    # spinup= length of time (years) to trim off start of results when model is 'spinning up'
    # kwargs


    # set/modify Seaborn defaults
    if len(plotstyle) == 0:
        plotstyle = {'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'axes.grid': False,
                     'grid.color': 'w'}

    sb.set() # reset default parameters first
    sb.set_style("ticks", plotstyle)

    # update the axes_style for seaborn with specified params
    sb.set_style("ticks", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    if not props:
        props = {'sresa1b': {'color': 'Tomato', 'zorder': -2, 'alpha': 0.5},
                 'sresa2': {'color': 'SteelBlue', 'zorder': -3, 'alpha': 0.5},
                 'sresb1': {'color': 'Yellow', 'zorder': -1, 'alpha': 0.5}}

    # initialize plot
    fig = plt.figure()
    #ax = fig.add_subplot(111, zorder=100)
    ax = fig.add_subplot(111)

    # global settings
    alpha = 0.5
    synthetic_timeper_color = '1.0'
    synthetic_timeper_alpha = 1 - (alpha * 0.4)


    for dfname in dfs.iterkeys():

        alpha = props[dfname]['alpha']
        color = props[dfname]['color']
        zorder = props[dfname]['zorder']

        try:
            dfs[dfname].mean(axis=1).plot(color=color, label=dfname, linewidth=1, zorder=zorder, ax=ax)
        except TypeError:
            print "Problem plotting timeseries. Check that spinup value was not entered for plotting after spinup results already discarded during aggregation."

        ax.fill_between(dfs[dfname].index, dfs[dfname].max(axis=1), dfs[dfname].min(axis=1),
                        alpha=alpha, color=color, edgecolor='k', linewidth=0.25, zorder=zorder*100)

    # rescale plot to ignore extreme outliers
    if clip_outliers:
        newylimit = ignore_outliers_in_yscale(dfs)
        if newylimit:
            ax.set_ylim(ax.get_ylim()[0], newylimit)


    # shade periods for which synthetic data were generated
    if len(Synthetic_timepers) > 0:
        for per in Synthetic_timepers:
            max, min = ax.get_ylim()
            ax.fill_between(per, [min] * len(per), [max] * len(per),
                            facecolor=synthetic_timeper_color, edgecolor=synthetic_timeper_color, alpha=synthetic_timeper_alpha,
                            linewidth=0, zorder=0)

    # make title
    make_title(ax, title)


    thousands_sep(ax) # fix the scientific notation on the y axis
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # prevent the shaded time periods of no data from overlapping the neatline
    # (spines is a dictionary)
    for side in ax.spines.iterkeys():
        ax.spines[side].set_zorder(200)

    plt.tight_layout()

    return fig, ax



def sb_violin_annual(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', color=['SteelBlue', 'Khaki'],
                     plotstyle={}, rcparams={}):

    sb.set() # reset default parameters first
    sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    # keyword arguments for plots inside violins
    # keywords are for pyplot.plot()
    inner_kws = {'linewidth': plt.rcParams['axes.linewidth'] * 1.5}

    dates = ['-'.join(map(str, per)) for per in compare_periods]

    fig = plt.figure()
    try:
        ax = sb.violinplot(boxcolumns, names=dates, color=color, inner_kws=inner_kws,
                           linewidth=plt.rcParams['axes.linewidth'])

        # plot the population of streamflows within each violin
        for i in range(len(compare_periods)):
            plt.plot([i+1]*len(boxcolumns[i]), boxcolumns[i].tolist(), markerfacecolor='k', linestyle='', marker='o')

        ax.axhline(y=baseline[0], xmin=0.05, xmax=0.95, color='r', linewidth=plt.rcParams['axes.linewidth']*2)

        # make title
        make_title(ax, title)

        thousands_sep(ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # set reasonable lower y limit for violins
        # (kernals can dip below zero even if data doesn't; inappropriate for strictly positive variables)
        minval = np.min([b.min() for b in boxcolumns])
        if minval < 0:
            ymin = np.min([b.min() for b in boxcolumns])
        else:
            ymin = ax.get_ylim()[0]
        ax.set_ylim(ymin, ax.get_ylim()[1])

    except:
        print sys.exc_info()
        ax = None

    plt.tight_layout()
    return fig, ax


def sb_box_annual(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', color=['SteelBlue', 'Khaki'],
                  plotstyle={}, rcparams={}, **kwargs):

    sb.set() # reset default parameters first
    sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    dates = ['-'.join(map(str, per)) for per in compare_periods]

    fig = plt.figure()

    ax = sb.boxplot(boxcolumns, names=dates, color=color,  **kwargs)

    ax.axhline(y=baseline[0], xmin=0.05, xmax=0.95, color='r', linewidth=plt.rcParams['axes.linewidth']*2)

    # make title
    make_title(ax, title)

    thousands_sep(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #ax.set_ylim(0, ax.get_ylim()[1])

    plt.tight_layout()
    return fig, ax


def sb_box_monthly(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', color=['SteelBlue', 'Khaki'],
                   xtick_freq=1,
                   plotstyle={}, rcparams={}, **kwargs):
    '''
    different method than annual because the boxes are grouped by month, with one tick per month
    '''

    sb.set() # reset default parameters first
    sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    dates = ['-'.join(map(str, per)) for per in compare_periods]

    # set box widths and positions so that they are grouped by month
    n_periods = len(dates)
    spacing = 0.1 # space between months
    boxwidth = (1 - 2 * spacing)/n_periods
    positions = []
    for m in range(12):
        for d in range(n_periods):
            position = 0.5 + m + spacing + (boxwidth * (d+0.5))
            positions.append(position)

    # make the box plot
    fig = plt.figure()
    ax = sb.boxplot(boxcolumns, positions=positions, widths=boxwidth, color=color, **kwargs)

    xmin, xmax = ax.get_xlim()
    l = xmax - xmin

    # transform the box positions to axes coordinates (one tuple per box)
    positions_t = []
    for i in range(len(positions)):
        positions_t.append((((positions[i] - 0.5*boxwidth - xmin)/l), (positions[i] + 0.5*boxwidth - xmin)/l))

    for i in range(len(baseline)):
        ax.axhline(baseline[i], xmin=positions_t[i][0], xmax=positions_t[i][1],
                   color='r', linewidth=plt.rcParams['axes.linewidth']*2)

    # clean up the axes
    thousands_sep(ax) # fix the scientific notation on the y axis

    # reset the ticks so that there is one per month (one per group)
    frequency = xtick_freq # 1 for every month, 2 for every other, etc.
    ticks = (np.arange(12) + 1)[frequency-1::frequency]
    ax.set_xticks(ticks)
    months = []
    for tick in ax.get_xticks():
        month = ReportFigures.month[tick]
        months.append(month)

    ax.set_xticklabels(months)

    # make title
    make_title(ax, title)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


    plt.tight_layout()
    return fig, ax
