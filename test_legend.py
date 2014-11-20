reload(cp)
plt.close('all')
Figs = cp.ReportFigures(mode, compare_periods, baseline_period, gcms, spinup,
                             results_folder, output_folder,
                             var_name_file,
                             timeseries_properties,
                             variables_table=vars,
                             synthetic_timepers=synthetic_timepers,
                             exclude=exclude)
fig = Figs.make_timeseries_legend()
plt.savefig('/Users/aleaf/Documents/BlackEarth/run3/plots_run3/timeseries_legend.pdf')
