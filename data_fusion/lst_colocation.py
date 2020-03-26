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

    out_latlon = pyproj.transform(modis_proj, latlon_proj, in_proj.values[:, 0], in_proj.values[:, 1])

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


def extract_merra_by_name(filename, dsname, m_lat, m_lon):
    """

    :param filename:
    :param dsname:
    :param m_lat:
    :param m_lon:
    :return:
    """
    h4_data = Dataset(filename)
    ds = h4_data[dsname][:]
    result = []
    for t in range(ds.shape[0]):
        # using current (self) index for DataFrame index and column
        temp = pd.DataFrame(data=h4_data[dsname][:][t])
        # selecting only data overlapping with MODIS to reduce data size: dataFrame.loc[<ROWS RANGE> , <COLUMNS RANGE>]
        temp = temp.loc[min(m_lat):max(m_lat), min(m_lon):max(m_lon)]
        result.append(temp)
    h4_data.close()

    return result


def extract_modis_by_name(filename, dsname, m_lat, m_lon):
    """

    :param filename:
    :param dsname:
    :param m_lat:
    :param m_lon:
    :return:
    """
    hdf = SD.SD(filename)
    result = []
    for ds in dsname:
        lst = hdf.select(ds)
        temp = lst.get()
        scale_factor = lst.attributes()['scale_factor']
        temp = temp * scale_factor
        # using merra2-based colocated index for DataFrame index and column, matching with merra2 dataset
        temp = pd.DataFrame(data=temp, index=m_lat, columns=m_lon)
        temp = temp.replace(0, np.nan)
        result.append(temp)

    return result


if __name__ == '__main__':

    '''getting files from path'''
    all_merra_files, me_all = get_files('/Volumes/Samsung_T5/IoT_HeatIsland_Data/MERRA2/', '*.nc4')
    all_modis_files, mo_all = get_files('/Volumes/Samsung_T5/IoT_HeatIsland_Data/MODIS/MOD/', '*.hdf')
    all_merra_files = np.sort(all_merra_files)
    all_mod_files = np.sort(all_modis_files)

    '''getting coordinate data'''
    mod_xy = pd.read_csv('/Volumes/Samsung_T5/IoT_HeatIsland_Data/MODIS/XYDim.csv')
    mod_xy = modis_conv(mod_xy)

    merra_data = Dataset(all_merra_files[0])
    merra_lat = np.array(merra_data['lat'][:])
    merra_lon = np.array(merra_data['lon'][:])

    '''colocation'''
    # using the index of merra2's coordinate to represent modis' coordinates
    co_mod_lat = colocation(merra_lat, mod_xy[1])
    co_mod_lon = colocation(merra_lon, mod_xy[0])

    '''collecting data from files'''
    # for i in range(min(me_all, mo_all)):
    for i in range(1):
        # a date matching function needed as naming system is different
        merra_file = all_merra_files[i]
        mod_file = all_mod_files[i]
        merra_SST = extract_merra_by_name(merra_file, 'TLML', co_mod_lat, co_mod_lon)
        mod_SST = extract_modis_by_name(mod_file, ['LST_Day_1km', 'LST_Night_1km'], co_mod_lat, co_mod_lon)
        merra_SST = [i.sort_index(ascending=False) for i in merra_SST]

        print(merra_SST, mod_SST)
