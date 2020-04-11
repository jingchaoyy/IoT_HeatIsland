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
import datetime
import os.path


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
        temp = pd.DataFrame(data=ds[t])
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


def match_date(mod_f_name, merra_p):
    """

    :param mod_f_name:
    :param merra_p:
    :return:
    """
    year = mod_f_name[9:16][:4]
    day = mod_f_name[9:16][4:]
    date = datetime.datetime.strptime(f'{year} {day}', '%Y %j')
    date = "{:4d}{:02d}{:02d}".format(date.year, date.month, date.day)

    # data_next = datetime.datetime.strptime(f'{year} {int(day) + 1}', '%Y %j')
    # data_next = "{:4d}{:02d}{:02d}".format(data_next.year, data_next.month, data_next.day)

    return merra_p + f'MERRA2_400.tavg1_2d_flx_Nx.{date}.nc4', date


def time_pattern_trans(base, trend):
    """
    transferring temporal pattern of a course grid to each associated finer grid
    :param base:
    :param trend:
    :return:
    """


if __name__ == '__main__':
    merra_path = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/MERRA2/'
    mod_path = '/Volumes/Samsung_T5/IoT_HeatIsland_Data/MODIS/MOD/'

    '''getting files from path'''
    all_merra_files, me_all = get_files(merra_path, '*.nc4')
    all_modis_files, mo_all = get_files(mod_path, '*.hdf')
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

    uni_lat = sorted(list(set(co_mod_lat)), reverse=True)
    uni_lon = sorted(list(set(co_mod_lon)))

    '''collecting data from files'''
    # for i in range(min(me_all, mo_all)):
    for f in range(min(len(all_merra_files), len(all_mod_files))):
        # a date matching function needed as naming system is different
        mod_file = all_mod_files[f]
        merra_file, cur_date = match_date(os.path.split(mod_file)[1], merra_path)

        merra_SST = extract_merra_by_name(merra_file, 'TLML', co_mod_lat, co_mod_lon)
        # mod_SST = extract_modis_by_name(mod_file, ['LST_Day_1km', 'LST_Night_1km'], co_mod_lat, co_mod_lon)
        merra_SST = [i.sort_index(ascending=False) for i in merra_SST]

        merra_SST_np = np.asarray([i.to_numpy() for i in merra_SST])
        # looping through each coor and get 24h timeseries data
        for lat in range(merra_SST_np.shape[1]):
            for lon in range(merra_SST_np.shape[2]):
                curr_lat = uni_lat[lat]
                curr_lon = uni_lon[lon]

                real_lat = merra_lat[curr_lat]
                real_lon = merra_lon[curr_lon]
                print(f'processing lat: {curr_lat} lon:{curr_lon}')

                merra_24h = merra_SST_np[:, lat, lon]
                datetime = [pd.to_datetime(cur_date + str(i), format='%Y%m%d%H') for i in range(0, 24)]
                df = pd.DataFrame(data=datetime, columns=['datetime'])
                df['lat'] = real_lat
                df['lon'] = real_lon
                df['temp'] = merra_24h

                path = f'/Volumes/Samsung_T5/IoT_HeatIsland_Data/MODIS/MOD_MERRA_Co_test/allMERRA/lat_{curr_lat}_lon_{curr_lon}.csv'
                if os.path.isfile(path):
                    df.to_csv(path, mode='a', header=False)
                else:
                    df.to_csv(path)

                print(df)

                # np.savetxt('', merra_24h, delimiter=",")
                # mod_within_10 = mod_SST[0].loc[curr_lat][curr_lon]
                # mod_within_22 = mod_SST[1].loc[curr_lat][curr_lon]
                #
                # vfunc = np.vectorize(time_pattern_trans)
                # vfunc(mod_within_10, mod_within_22, merra_24h)
