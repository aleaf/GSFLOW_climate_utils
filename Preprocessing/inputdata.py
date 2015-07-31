__author__ = 'aleaf'

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, pyproj
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.collections import PatchCollection
from GISio import shp2df


class GDPfiles:

    def __init__(self, dir, variable='tmin', scenarios=['20c3m', 'early', 'late'],
                 shapefile='', shapefile_hru_col='nhru'):

        tminfiles = [os.path.join(dir, f) for f in os.listdir(dir) if variable in f]

        if shapefile is not None:
            shp = shp2df(shapefile)
            try:
                shp.sort(shapefile_hru_col, inplace=True)
                self.geometry = shp[['geometry']]
                self.geometry.index = np.arange(len(shp)) + 1
            except Exception, e:
                print Exception, e
                print '\nPlease supply an index field relating shapefile geometries ' \
                      'to columns in GDP data (e.g. hru number)'


        df = self.geometry.copy()
        for f in tminfiles:
            dff = pd.read_csv(f, skiprows=3, header=None, index_col=0, parse_dates=True)
            df[f] = dff.groupby(lambda x: x.year).mean().mean(axis=0)
        self.df = df.copy()

        scenario_avg = self.geometry.copy()
        for scen in scenarios:
            scenario_avg[scen] = df[[f for f in tminfiles if scen in f]].mean(axis=1)
        self.scenario_avg = scenario_avg.copy()

class hrumap:

    def __init__(self, shapefile='', shapefile_hru_col='nhru'):

        shp = shp2df(shapefile)
        try:
            shp.sort(shapefile_hru_col, inplace=True)
            self.geometry = shp['geometry']
            self.geometry.index = shp[shapefile_hru_col].values
        except Exception, e:
            print Exception, e
            print '\nPlease supply an index field relating shapefile geometries ' \
                  'to columns in GDP data (e.g. hru number)'

        self.hrus = None
        self.patches = None

    def make_patches(self, simplify=100):

        from descartes import PolygonPatch

        if simplify > 0:
            self.geometry = pd.Series([g.simplify(simplify) for g in self.geometry], 
                                       index=self.geometry.index)
        hrus = []
        patches = []
        for i, g in self.geometry.iteritems():
            if g.type != 'MultiPolygon':
                hrus.append(i)
                patches.append(PolygonPatch(g))
            else:
                for part in g.geoms:
                    hrus.append(i)
                    patches.append(PolygonPatch(part))
        self.hrus = hrus
        self.patches = patches

    def make_maps(self, df, columns=None, cbar_label='values',
                  outpdf=None, figsize=(8.5, 11),
                  cbar=False, clim=(), cmap='jet',
                  simplify_patches=100, ax=None):
        """Make a map of input data values

        df : dataframe
            Data to map. Must have geometry column of features.
            Remaining columns have values for gcm-scenario-periods.
        column : str
            Column(s) in dataframe with values to map. If default of None, map all columns.
        """
        if columns == None:
            columns = df.columns

        if not self.patches:
            self.make_patches(simplify=simplify_patches)

        if not isinstance(cbar_label, dict):
            if isinstance(cbar_label, list):
                cbar_label = {columns[i]:label for i, label in enumerate(cbar_label)}
            else:
                cbar_label = {c:cbar_label for c in columns}

        axes = []
        for c in columns:
            print c
            colors = [df[c][ind] for ind in self.hrus]
            pc = PatchCollection(self.patches, cmap=cmap, lw=0)
            pc.set_array(np.array(colors))
            if len(clim) > 0:
                pc.set_clim(clim[0], clim[1])

            if ax is None:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, aspect='equal')

            ax.add_collection(pc)
            ax.autoscale_view()
            if cbar:
                plt.colorbar(pc, label=cbar_label[c])
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.set_ylabel('Northing')
            ax.set_xlabel('Easting')
            if outpdf is not None:
                outpdf.savefig()
            else:
                axes.append(ax)
        if len(columns) > 1:
            return axes
        else:
            return ax

    def map_grid(self, df, columns=None,
                 titles=None, cbar_label='',
                  outpdf=None, figsize=(8.5, 11),
                  nrows_ncols=(1, 2),
                  cbar=False, clim=(), cmap='jet',
                  simplify_patches=100, ax=None):
        """Make a map of input data values

        df : dataframe
            Data to map. Must have geometry column of features.
            Remaining columns have values for gcm-scenario-periods.
        column : str
            Column(s) in dataframe with values to map. If default of None, map all columns.
        """
        if columns == None:
            columns = df.columns

        if not self.patches:
            self.make_patches(simplify=simplify_patches)

        #titles = _parse_label_arg(columns, titles)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11, 8.5),
                         sharex=True, sharey=True)

        for i, c in enumerate(columns):

            ax = axes.flat[i]
            colors = [df[c][ind] for ind in self.hrus]
            pc = PatchCollection(self.patches, cmap=cmap, lw=0)
            pc.set_array(np.array(colors))
            if len(clim) > 0:
                pc.set_clim(clim[0], clim[1])

            ax.add_collection(pc)
            ax.autoscale_view()
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))

        plt.tight_layout()
        fig.colorbar(pc, ax=axes.ravel().tolist(), pad=0.01, label=cbar_label)

        return axes.flat

def _parse_label_arg(columns, arg):

    if not isinstance(arg, dict):
        if isinstance(arg, list):
            argdict = {columns[i]: label for i, label in enumerate(arg)}
        else:
            argdict = {c: arg for c in columns}
    elif len(arg) == 0:
        argdict = {c: arg for c in columns}
    else:
        argdict = arg
    return argdict