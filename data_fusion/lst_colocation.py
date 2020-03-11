"""
Created on  3/11/20
@author: Jingchao Yang
"""
import os
import glob
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from pyhdf import SD
import pyproj


def modis_conv(in_proj):
    """
    https://all-geo.org/volcan01010/2012/11/change-coordinates-with-pyproj/
    https://spatialreference.org/ref/sr-org/modis-sinusoidal/proj4/
    Converting projected modis XDim, YDim to lonlat

    :param in_proj:
    :return:
    """
    modis_proj = pyproj.CRS("+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=km +no_defs")
    latlon_proj = pyproj.CRS(proj='latlong')

    y1 = [4447.802]
    x1 = [-11119.51]
    out_latlon = pyproj.transform(modis_proj, latlon_proj, x1, y1)

    return out_latlon


def colocation(refx, x):
    """
    mapping to a reference coordinate system (self defined)

    :param refx:
    :param x:
    :return:
    """
    refx = np.array(refx)
    x = np.array(x)
    loc = np.zeros(len(x), dtype='int')

    for i in range(len(x)):
        ix = x[i]
        ind = ((refx - ix) <= 0).nonzero()[0]
        if len(ind) == 0:
            loc[i] = -1
        else:
            loc[i] = ind[-1]

    return loc


def get_files(dir, ext):
    """

    :param dir:
    :param ext:
    :return:
    """
    allfiles = []
    os.chdir(dir)
    for file in glob.glob(ext):
        allfiles.append(dir + file)
    return allfiles, len(allfiles)


def extract_merra_by_name(filename, dsname):
    """

    :param filename:
    :param dsname:
    :return:
    """
    h4_data = Dataset(filename)
    ds = h4_data[dsname][:]
    lat = np.array(h4_data['lat'][:])
    lon = np.array(h4_data['lon'][:])
    result = []
    for t in range(ds.shape[0]):
        temp = pd.DataFrame(data=h4_data[dsname][:][t], index=lat, columns=lon)
        result.append(temp)
    h4_data.close()
    return result


def extract_modis_by_name(filename, dsname):
    """

    :param filename:
    :param dsname:
    :return:
    """
    hdf = SD.SD(filename)
    LST_day = hdf.select(dsname)
    temp_d = LST_day.get()
    scale_factor = LST_day.attributes()['scale_factor']
    temp_d = temp_d * scale_factor

    return temp_d


if __name__ == '__main__':

    all_merra_files, me_all = get_files('/Volumes/Samsung_T5/IoT_HeatIsland_Data/MERRA2/', '*.nc4')
    all_merra_files = np.sort(all_merra_files)

    all_modis_files, mo_all = get_files('/Volumes/Samsung_T5/IoT_HeatIsland_Data/MODIS/', '*.hdf')
    all_mod_files = np.sort(all_modis_files)

    for i in range(min(me_all, mo_all)):
        merra_file = all_merra_files[i]
        mod_file = all_mod_files[i]
        merra_SST = extract_merra_by_name(merra_file, 'TLML')
        mod_SST = extract_modis_by_name(mod_file, 'LST_Day_1km')

        print(merra_SST, mod_SST)
