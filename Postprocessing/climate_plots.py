__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.patheffects as PathEffects
from matplotlib.colors import LinearSegmentedColormap
import textwrap
import climate_stats as cs
import GSFLOW_utils as GSFu
from Figures import ReportFigures

rf = ReportFigures()
rf.set_style()

'''
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
'''
'''
# set/modify global Seaborn defaults
# update any overlapping parameters in the seaborn 'paper' style with the custom values above
sb.set_context("paper", rc=newparams)
'''



class ReportFigures():

    # figure sizes (based on 6 picas per inch, see USGS Illustration Standards Guide, p 34
    default_aspect = 6 / 8.0 # h/w
    tall_aspect = 7 / 8.0
    singlecolumn_width = 21/6.0
    doublecolumn_width = 42/6.0

    # title wraps
    singlecolumn_title_wrap = 50
    doublecolumn_title_wrap = 120

    # month abbreviations
    month = {1: 'Jan.', 2: 'Feb.', 3: 'Mar.', 4: 'Apr.', 5: 'May', 6: 'June',
             7: 'July', 8: 'Aug.', 9: 'Sept.', 10: 'Oct.', 11: 'Nov.', 12: 'Dec.'}

    # no negative values on yaxis of plots
    ymin0 = False

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
                 box_colors=['SteelBlue', 'Khaki'],
                 stat_summary=True):

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
        self.dates = ['-'.join(map(str, per)) for per in self.compare_periods]
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

        # make a summary file of statistics by variable and period (for checking box/violin plots)
        if stat_summary:
            self.stat_summary = True
            self.ofp = open('summary_stats.csv', 'w')
            self.ofp.write('variable,period,stat,max,upperQ,median,lowerQ,min,baseline\n')
        else:
            self.stat_summary=False


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


    def plot_info(self, var, stat, plottype, quantile=None, normalize_to_baseline=False):
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

        if normalize_to_baseline:
            ylabel = 'Percent change relative to baseline period'
        return title, xlabel, ylabel, calc


    def make_box(self, csvs, var, stat, quantile=None, normalize_to_baseline=False):

        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'box', quantile=quantile,
                                                     normalize_to_baseline=normalize_to_baseline)

        # calculate montly means for box plot
        boxcolumns, baseline = cs.period_stats(csvs, self.compare_periods, stat, self.baseline_period,
                                               calc=calc, quantile=quantile, normalize_to_baseline=normalize_to_baseline)



        # make box plot
        if 'month' in stat:

            # settings to customize Seaborn "ticks" style (i.e. turn off grid)
            rcparams = {'figure.figsize': (self.doublecolumn_width, self.doublecolumn_width * self.tall_aspect),
                        'lines.linewidth': 0.5}

            fig, ax = box_monthly(boxcolumns, baseline, self.compare_periods, ylabel,
                                     color=self.box_colors, plotstyle=self.plotstyle,
                                     rcparams=rcparams,
                                     fliersize=3, linewidth=0.5)
            self.figure_title(ax, title, wrap=self.doublecolumn_title_wrap)

        else:
            # settings to customize Seaborn "ticks" style (i.e. turn off grid)
            rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.tall_aspect),
                        'lines.linewidth': 0.5}

            fig, ax = box_annual(boxcolumns, baseline, self.compare_periods, ylabel,
                                    color=self.box_colors, plotstyle=self.plotstyle,
                                    rcparams=rcparams,
                                    fliersize=plt.rcParams['figure.figsize'][0] * 1.5, linewidth=0.5)

            self.figure_title(ax, title, wrap=self.singlecolumn_title_wrap)

        self.axes_numbering(ax)

        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'box', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()


    def make_timeseries(self, csvs, var, stat, quantile=None, baseline=False, baseline_text=False):

        # Set plot titles and ylabels
        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'timeseries', quantile=quantile)

        # calculate annual means
        dfs = cs.annual_timeseries(csvs, self.gcms, self.spinup, stat, calc=calc, quantile=quantile)

        if baseline:
            bl = pd.Panel(dfs).ix[:, str(self.baseline_period[0]):str(self.baseline_period[1])]\
                .mean().mean().mean()
        else:
            bl = None
        # settings to customize Seaborn "ticks" style (i.e. turn off grid)
        rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.default_aspect),
                    'axes.grid': False,
                    'grid.linestyle': ''}

        # make 'fill_between' timeseries plot with  mean and min/max for each year
        fig, ax = timeseries(dfs, ylabel=ylabel, props=self.timeseries_properties, Synthetic_timepers=self.synthetic_timepers,
                             rcparams=rcparams, baseline=bl)

        self.figure_title(ax, title, wrap=self.singlecolumn_title_wrap)
        #self.axes_numbering(ax)

        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'timeseries', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()

    def make_timeseries_hexbin(self, csvs, var, stat, quantile=None, baseline=False, baseline_text=False):

        # Set plot titles and ylabels
        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'timeseries', quantile=quantile)

        # calculate annual means
        dfs = cs.annual_timeseries(csvs, self.gcms, self.spinup, stat, calc=calc, quantile=quantile)

        if baseline:
            bl = pd.Panel(dfs).ix[:, str(self.baseline_period[0]):str(self.baseline_period[1])]\
                .mean().mean().mean()
        else:
            bl = None
        # settings to customize Seaborn "ticks" style (i.e. turn off grid)
        rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.default_aspect),
                    'axes.grid': False,
                    'grid.linestyle': ''}

        # make 'fill_between' timeseries plot with  mean and min/max for each year
        fig, ax = timeseries_hexbin(dfs, ylabel=ylabel, props=self.timeseries_properties,
                                    Synthetic_timepers=self.synthetic_timepers,
                                    baseline=bl)

        self.figure_title(ax, title, wrap=self.singlecolumn_title_wrap)
        #self.axes_numbering(ax)

        plt.tight_layout() # call this again so that title doesn't get cutoff

        outfile = self.name_output(var, stat, 'timeseries_hexbin', quantile)
        fig.savefig(outfile, dpi=300)
        plt.close()

    def make_violin(self, csvs, var, stat, quantile=None):

        # Set plot titles and ylabels
        title, xlabel, ylabel, calc = self.plot_info(var, stat, 'box', quantile=quantile)

        # calcualte period statistics for violins
        boxcolumns, baseline = cs.period_stats(csvs, self.compare_periods, stat, self.baseline_period,
                                               calc=calc, quantile=quantile)

        # write summary information (for checking plots)
        if self.stat_summary:
            for i, d in enumerate(self.dates):
                self.ofp.write('{},{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'
                               .format(var, d, stat,
                                       np.max(boxcolumns[i]),
                                       boxcolumns[i].quantile(q=0.75),
                                       boxcolumns[i].quantile(q=0.50),
                                       boxcolumns[i].quantile(q=0.25),
                                       np.min(boxcolumns[i]),
                                       baseline[i]))


        # settings to customize Seaborn "ticks" style (i.e. turn off grid)
        rcparams = {'figure.figsize': (self.singlecolumn_width, self.singlecolumn_width * self.tall_aspect),
                    'lines.linewidth': 0.5,
                    'lines.markersize': self.singlecolumn_width}

        # make violin plot
        fig, ax = sb_violin_annual(boxcolumns, baseline, self.compare_periods, ylabel,
                                   color=self.box_colors, plotstyle=self.plotstyle,
                                   rcparams=rcparams)

        self.figure_title(ax, title, wrap=self.singlecolumn_title_wrap)

        # suppress negative numbers on yaxis if the data aren't negative
        if not np.min([np.min(boxcolumns), np.min(baseline)]) < 0:
            self.ymin0 = True

        self.axes_numbering(ax)
        self.ymin0 = False # go back to default

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
        handles.append(plt.Line2D(range(10), range(10), color='r', linewidth=0.5))
        labels.append('Baseline conditions ({}-{})'.format(self.baseline_period[0], self.baseline_period[1]))
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=0.5))
        labels.append('Boxes represent quartiles;'.format(self.baseline_period[0], self.baseline_period[1]))
        handles.append(plt.Line2D(range(10), range(10), color='k', linewidth=0.5))
        labels.append('Whiskers represent 1.5x the interquartile range')
        handles.append(plt.scatter(1, 1, c='k', s=6, marker='D'))
        labels.append('Outliers')

        figlegend = plt.figure(figsize=(3, 2))
        lg = figlegend.legend(handles, labels, title='EXPLANATION', loc='center',
                              borderpad=1)

        # update legend title size and font size to conform
        plt.setp(lg.get_title(), fontsize=self.legend_titlesize)


        outfile = os.path.join(self.output_base_folder, 'box_legend.pdf')
        figlegend.savefig(outfile, dpi=300)
        plt.close('all')

    def make_timeseries_hexbin_legend(self):

        fig = plt.figure()

        plt.rcParams.update({'font.family': self.legend_font,
                             'legend.fontsize': self.legend_fontsize})

        handles=[]
        labels=[]
        for scn in ['sresa1b', 'sresa2', 'sresb1']:
            handles.append(plt.Line2D(range(10), range(10),
                                      color=self.timeseries_properties[scn]['color'],
                                      linewidth=self.timeseries_properties[scn]['lw']))
            labels.append(scn)
            handles[-1].set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
        figlegend = plt.figure(figsize=(3, 2))
        lg = figlegend.legend(handles, labels, title='EXPLANATION', loc='center',
                              borderpad=1)

        # update legend title size and font size to conform
        plt.setp(lg.get_title(), fontsize=self.legend_titlesize)


        outfile = os.path.join(self.output_base_folder, 'ts_hexbin_legend.pdf')
        figlegend.savefig(outfile, dpi=300)
        plt.close('all')

    def make_timeseries_legend(self, baseline=False):

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

            y = 0.5
            if baseline:
                y=0.67
                # lines to represent means
                l = grid[i].axhline(y=0.33, xmin=0, xmax=1, color='k', alpha=0.33,
                                    lw=1, zorder=100)

            # lines to represent means
            l = grid[i].axhline(y=y, xmin=0, xmax=1, color=self.timeseries_properties[scen[i]]['color'],
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
        grid[0].annotate('Screening indicates model spin-up period',
                         xy=(0.18, 0), xycoords='data', xytext=(0.5, -0.35), ha='left', va='center',
                         arrowprops=dict(arrowstyle='-', linewidth=0.5, relpos=(0, .4)))


        # Labels for max/mean/min
        grid[i].text(1.2, 1, 'Maximum', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)
        grid[i].text(1.2, y, 'Mean from General Circulation Models', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)
        grid[i].text(1.2, 0, 'Minimum', ha='left', va='center',
                     transform=grid[i].transAxes, family=self.legend_font, fontsize=self.legend_fontsize)
        if baseline:
            grid[i].text(1.2, 0.33, 'Mean for baseline period of {}-{}'.format(*list(self.baseline_period)),
                         ha='left', va='center',
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


    def figure_title(self, ax, title, zorder=200, wrap=50):
        wrap = wrap
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


    def axes_numbering(self, ax):
        '''
        Implement these requirements from USGS standards, p 16
        * Use commas in numbers greater than 999
        * Label zero without a decimal point
        * Numbers less than 1 should consist of a zero, a decimal point, and the number
        * Numbers greater than or equal to 1 need a decimal point and trailing zero only where significant figures dictate
        '''

        # enforce minimum value of zero on y axis if data is positive
        if self.ymin0 and ax.get_ylim()[0] < 0:
            ax.set_ylim(0, ax.get_ylim()[1])


        # so clunky, but this appears to be the only way to do it
        if -10 > ax.get_ylim()[0] or ax.get_ylim()[1] > 10:
            fmt = '{:,.0f}'
        elif -10 <= ax.get_ylim()[0] < -1 or 1 < ax.get_ylim()[1] <= 10:
            fmt = '{:,.1f}'
        elif -1 <= ax.get_ylim()[0] < -.1 or .1 < ax.get_ylim()[1] <= 1:
            fmt = '{:,.2f}'
        else:
            fmt = '{:,.2e}'

        def format_axis(y, pos):
            y = fmt.format(y)
            return y

        ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(format_axis))

        # correct for edge cases of upper ylim == 1, 10
        # and fix zero to have no decimal places
        def fix_decimal(ticks):
            # ticks are a list of strings
            if float(ticks[-1]) == 10:
                ticks[-1] = '10'
            if float(ticks[-1]) == 1:
                ticks[-1] = '1.0'
            for i, v in enumerate(ticks):
                if float(v) == 0:
                    ticks[i] = '0'
            return ticks

        text_xlabels = False
        text_ylabels = False
        try:
            # only proceed if labels are numbers (there might be a better way to do this)
            [float(l._text.replace(u'\u2212', '-')) for l in ax.get_xticklabels()]
            #newxlabels = fix_decimal([fmt.format(float(l._text.replace(u'\u2212', '-'))) for l in ax.get_xticklabels()])
            ax.get_yticks()
            newxlabels = fix_decimal([fmt.format(t) for t in ax.get_yticks()])
            ax.set_xticklabels(newxlabels)

        except:
            text_xlabels = True
            pass

        try:
            # only proceed if labels are numbers (there might be a better way to do this)
            [float(l._text.replace(u'\u2212', '-')) for l in ax.get_yticklabels()]
            ax.get_yticks()
            newylabels = fix_decimal([fmt.format(t) for t in ax.get_yticks()])
            #newylabels = fix_decimal([fmt.format(float(l._text.replace(u'\u2212', '-'))) for l in ax.get_yticklabels()])
            ax.set_yticklabels(newylabels)
        except:
            text_ylabels = True
            pass

        if text_xlabels and text_ylabels:
            print "Warning: Tick labels on both x and y axes are text. Or the canvas may not be drawn, in which" \
                  "case the tick formatter won't work. Check the code for generating the plot."


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
               plotstyle={}, rcparams={}, baseline=None, baseline_text=None):
    """
    Makes a timeseries plot from dataframe(s) containing multiple timeseries of the same phenomena
    (e.g. multiple GCM realizations of future climate
    Plots the mean of all columns, enveloped by the min, max values of each row (as a fill-between plot)
    dfs : dict of dataframes to plot
    One dataframe per climate scenario; dataframes should have datetime indices

    """

    # set/modify Seaborn defaults
    if len(plotstyle) == 0:
        plotstyle = {'xtick.direction': 'in',
                     'ytick.direction': 'in',
                     'xtick.minor.size': 0,
                     'axes.grid': False,
                     'grid.linewidth': 0}

    #sb.set() # reset default parameters first
    #sb.set_style("ticks", plotstyle)

    # update the axes_style for seaborn with specified params
    #sb.set_style("ticks", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

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

    if baseline is not None:
        ax.axhline(baseline, c='k', lw=0.5, alpha=0.33, zorder=10)
        if baseline_text is not None:
            ymin, ymax = ax.get_ylim()
            y = (baseline - ymin) / (ymax - ymin)
            ax.text(0.5, y, baseline_text,
                    fontsize=6,
                    ha='right', va='bottom',
                    transform=ax.transAxes, zorder=10)
    # make title
    make_title(ax, title)


    #thousands_sep(ax) # fix the scientific notation on the y axis
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # prevent the shaded time periods of no data from overlapping the neatline
    # (spines is a dictionary)
    for side in ax.spines.iterkeys():
        ax.spines[side].set_zorder(200)

    plt.tight_layout()

    return fig, ax

def timeseries_hexbin(dfs, ylabel='', props=None, Synthetic_timepers=[],
               clip_outliers=True, xlabel='', title='',
               plotstyle={}, rcparams={}, baseline=None, baseline_text=None,
               **kwargs):
    """
    Makes a timeseries plot from dataframe(s) containing multiple timeseries of the same phenomena
    (e.g. multiple GCM realizations of future climate
    Plots the mean of all columns, enveloped by the min, max values of each row (as a fill-between plot)
    dfs : dict of dataframes to plot
    One dataframe per climate scenario; dataframes should have datetime indices

    """

    if not props:
        props = {'sresa1b': {'color': '0.0', 'zorder': -1, 'alpha': 1, 'linestyle': '-'},
                 'sresa2': {'color': '0.25', 'zorder': -3, 'alpha': 1, 'linestyle': '-'},
                 'sresb1': {'color': '0.5', 'zorder': -2, 'alpha': 1, 'linestyle': '-'}}

    # initialize plot
    fig, ax = plt.subplots()

    # global settings
    alpha = 0.5
    synthetic_timeper_color = '1.0'
    synthetic_timeper_alpha = 1 - (alpha * 0.4)
    # hexbin has int64 index
    Synthetic_timepers = [np.unique([dt.year for dt in per]) for per in Synthetic_timepers]

    df = pd.Panel(dfs).to_frame()
    stacked = df.stack().reset_index()
    stacked['year'] = [ts.year for ts in pd.to_datetime(stacked.major)]
    dfm = df.groupby(level=0).mean()
    nyears = len(stacked.year.unique())

    g = LinearSegmentedColormap.from_list('moregray', ((0.9, 0.9, 0.9), (0,0,0)), N=1000, gamma=1.0)

    default_kwargs = {'gridsize': (nyears, 40), 'cmap': g, 'mincnt': 1, 'zorder': -10}
    default_kwargs.update(kwargs)

    ax = stacked.plot(ax=ax, kind='hexbin', x='year', y=0, colorbar=False, **default_kwargs)
    hb = ax.get_children()[0] # assuming the polycollection will always be at 0
    cb = plt.colorbar(hb, label='Bin counts', pad=0.01)

    for scn, p in props.items():
        l = plt.plot(dfm.index.year, dfm[scn].tolist(),
                     zorder=p['zorder'], label=scn,
                     lw=p.get('lw', 1), color=p['color'], ls=p.get('linestyle', '-'))
        l[0].set_path_effects([PathEffects.withStroke(linewidth=3, foreground="k")])
    # shade periods for which synthetic data were generated
    if len(Synthetic_timepers) > 0:
        for per in Synthetic_timepers:
            max, min = ax.get_ylim()
            ax.fill_between(per, [min] * len(per), [max] * len(per),
                            facecolor=synthetic_timeper_color, edgecolor=synthetic_timeper_color, alpha=synthetic_timeper_alpha,
                            linewidth=0, zorder=0)

    if baseline is not None:
        ax.axhline(baseline, c='k', lw=0.5, alpha=0.5, zorder=20)
        if baseline_text is not None:
            ymin, ymax = ax.get_ylim()
            y = (baseline - ymin) / (ymax - ymin)
            ax.text(0.5, y, baseline_text,
                    fontsize=6,
                    ha='right', va='bottom',
                    transform=ax.transAxes, zorder=10)
    # make title
    make_title(ax, title)

    #thousands_sep(ax) # fix the scientific notation on the y axis
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # prevent the shaded time periods of no data from overlapping the neatline
    # (spines is a dictionary)
    for side in ax.spines.iterkeys():
        ax.spines[side].set_zorder(200)

    plt.tight_layout()

    return fig, ax


def sb_violin_annual(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', color=['SteelBlue', 'Khaki'],
                     plotstyle={}, rcparams={}, ymin0=True):
    '''
    #sb.set() # reset default parameters first
    #sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    #sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

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
        #ax.set_autoscale_on(False)

        for i in range(len(compare_periods)):
            ax.plot([i+1]*len(boxcolumns[i]), boxcolumns[i].tolist(), markerfacecolor='k', linestyle='', marker='o', axes=ax)

        ax.axhline(y=baseline[0], xmin=0.05, xmax=0.95, color='r', linewidth=plt.rcParams['axes.linewidth']*2)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    except:
        print sys.exc_info()

    # reset yaxis to zero if the violin dips below zero and there are no zero data
    if ymin0 and not np.min([np.min(boxcolumns), np.min(baseline)]) < 0 and ax.get_ylim()[0] < 0:
        ax.set_ylim(0, ax.get_ylim()[1])

    plt.tight_layout()
    return fig, ax
    '''
    pass

def sb_box_annual(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', color=['SteelBlue', 'Khaki'],
                  plotstyle={}, rcparams={}, ax=None, **kwargs):

    #sb.set() # reset default parameters first
    #sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    #sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    dates = ['-'.join(map(str, per)) for per in compare_periods]

    if ax is None:
        fig = plt.figure()

    ax = sb.boxplot(boxcolumns, names=dates, color=color,  ax=ax, **kwargs)

    ax.axhline(y=baseline[0], xmin=0.05, xmax=0.95, color='r', linewidth=plt.rcParams['axes.linewidth']*2)

    # make title
    make_title(ax, title)

    #thousands_sep(ax)
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

    #sb.set() # reset default parameters first
    #sb.set_style("whitegrid", plotstyle)

    # update the axes_style for seaborn with specified params
    #sb.set_style("whitegrid", sb.axes_style(rc=rcparams)) # apply additional custom styles for this plot

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

    #kwargs.update({'positions': positions})
    # make the box plot
    fig = plt.figure()
    ax = sb.boxplot(boxcolumns, width=boxwidth, palette=color, **kwargs) #positions=positions,

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
    #thousands_sep(ax) # fix the scientific notation on the y axis

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

def box_annual(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', colors=['SteelBlue', 'Khaki'],
                  plotstyle={}, rcparams={}, **kwargs):

    # update rcparams (because some style adjustments aren't handled by seaborn!)
    plt.rcParams.update(plotstyle)
    plt.rcParams.update(rcparams)

    dates = ['-'.join(map(str, per)) for per in compare_periods]

    fig, ax = plt.subplots()

    box = ax.boxplot(boxcolumns, widths=0.6, labels=dates, patch_artist=True,
                    whiskerprops={'ls':'-', 'c': 'k', 'lw': 0.5},
                    medianprops={'color':'k', 'zorder': 11},
                    flierprops={'marker': 'D', 'mfc': 'k', 'ms': 4}
                    )

    for patch, color in zip(box['boxes'], colors):
        patch.set_linewidth(0.5)
        patch.set_facecolor(color)
        patch.set_zorder(10)

    ax.axhline(y=baseline[0], xmin=0.05, xmax=0.95, color='r', linewidth=1.0, zorder=12)

    # make title
    make_title(ax, title)

    #thousands_sep(ax)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    #ax.set_ylim(0, ax.get_ylim()[1])

    plt.tight_layout()
    return fig, ax

def box_monthly(boxcolumns, baseline, compare_periods, ylabel, xlabel='', title='', colors=['SteelBlue', 'Khaki'] * 12,
                xtick_freq=1,
                plotstyle={}, rcparams={}, **kwargs):

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

    fig, ax = plt.subplots()
    ax.grid(True, which='major', axis='y', color='0.65',linestyle='-', zorder=-1)

    box = ax.boxplot(boxcolumns.values, widths=boxwidth, positions=positions, patch_artist=True,
                     whiskerprops={'ls':'-', 'c': 'k', 'lw': 0.5},
                     medianprops={'color':'k', 'zorder': 11},
                     flierprops={'marker': 'D', 'mfc': 'k', 'ms': 2}
                     )

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_zorder(10)

    # draw the baselines
    xmin, xmax = ax.get_xlim()
    l = xmax - xmin
    positions_t = []
    for i in range(len(positions)):
        positions_t.append((((positions[i] - 0.5*boxwidth - xmin)/l), (positions[i] + 0.5*boxwidth - xmin)/l))

    for i in range(len(baseline)):
        ax.axhline(baseline[i], xmin=positions_t[i][0], xmax=positions_t[i][1],
                   color='r', linewidth=1, zorder=12)

    frequency = xtick_freq # 1 for every month, 2 for every other, etc.
    ticks = (np.arange(12) + 1)[frequency-1::frequency]
    ax.set_xticks(ticks)
    months = []
    for tick in ax.get_xticks():
        month = ReportFigures.month[tick]
        months.append(month)
    ax.set_xticklabels(months)

    ax.set_xticklabels(months)

    # make title
    make_title(ax, title)

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)


    plt.tight_layout()
    return fig, ax