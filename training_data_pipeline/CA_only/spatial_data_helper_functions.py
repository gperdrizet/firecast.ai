''' Collection of helper functions for working with spatial data
used in geospatial filtering of weather data and regridding of fire
data to match weather data'''

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import CA_only.config as config


def load_polygon(shapefile):
    '''Loads US Census Bureau state boarders shapefile,
    returns California boarders as a shaply
    polygon object'''

    gdf = gpd.read_file(shapefile)
    california = gdf[gdf['NAME'] == 'California']
    multipoly = california.loc[16, 'geometry']
    california = multipoly[-1]

    return california


# need to load California polygon from disk
# do this once and make avalible as global so that
# we don't load it every time the function
# 'is_ca_point' function is called
CALIFORNIA = load_polygon(config.US_STATES_SHAPEFILE)
EMPTY = pd.Series([np.nan, np.nan])
EMPTY.index = ['lon', 'lat']


def is_ca_point(point):
    '''Takes point and returns point if in US'''

    coord = Point(point['lon'], point['lat'])
    if coord.within(CALIFORNIA):
        return point

    return EMPTY


def keep_ca_points(data: 'DataFrame') -> 'DataFrame':
    '''Takes a dataframe containing and uses apply to
    run a function on it. Called by parallelize.'''

    keepers = data.apply(is_ca_point, axis=1)

    return keepers


def spatial_filter_coarse(data: 'DataFrame', bounding_box: list) -> 'DataFrame':
    '''Takes dataframe containing lat, lon columns. Returns
    only rows which fall inside California bounding box
    coordinates.'''

    data = data.loc[(data['lat'] >= bounding_box[0]) &
                    (data['lat'] <= bounding_box[1])]

    data = data.loc[(data['lon'] >= bounding_box[2]) &
                    (data['lon'] <= bounding_box[3])]

    return data


def spatial_filter_fine(data):
    '''Takes dataframe and does innerjoin with US geospatial bins'''

    keeper_geospatial_bins = pd.read_parquet(
        config.TARGET_GEOSPATIAL_BINS_FILE)

    keepers = data.merge(keeper_geospatial_bins, left_on=['lat', 'lon'],
                         right_on=['lat', 'lon'], how='inner')

    return keepers
