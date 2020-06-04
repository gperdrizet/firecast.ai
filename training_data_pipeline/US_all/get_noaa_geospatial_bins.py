'''Uses a sample weather data file and the boarders of the continental
US as a polygon object to determine which geospatial points in the 
weather data are 'keepers'. I.e discard weather data from lat, lon
points over the ocean. The 'keeper' points will then be uset to 
fine filter all of the weather data.
'''

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import config

logging.basicConfig(
    filename=config.GET_TARGET_GEOSPATIAL_BINS_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

def check_bins(point, polygon, empty):
    '''Takes lat, lon point and polygon and returns point if in polygon,
    or empty lat, lon row otherwise'''
    coord = Point(point['lon'], point['lat'])
    if coord.within(polygon):
        return point

    return empty

def main():
    '''Reads weather data and polygon from disk. Writes weather data points
    in polygon to disk
    '''

    # load sample weather data
    sample_file = (config.DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR + config.SAMPLE_WEATHER_DATA_FILE)
    sample_weather_data = pd.read_parquet(sample_file)

    # keep only the first timepoint to reduce the size of the data
    t_zero = sample_weather_data['time'][0]
    sample_weather_data = sample_weather_data[sample_weather_data['time'] == t_zero]
    sample_weather_data.drop(['time', 'air.sfc'], axis=1, inplace=True)

    # read in the United States polygon
    gdf = gpd.read_file(config.SHAPEFILE)
    multipoly = gdf['geometry']
    polygon = multipoly[0][config.TARGET_POLYGON]

    # define empty lat, lon row to replace out of area points with
    EMPTY = pd.Series([np.nan, np.nan])
    EMPTY.index = ['lon', 'lat']

    # check to see if each geospatial weather bin the the sample data is
    # in the continental US or not
    keepers = sample_weather_data.apply(check_bins, args=(polygon, EMPTY), axis=1)

    # write keepers to disk
    keepers.to_parquet(config.TARGET_GEOSPATIAL_BINS_FILE)

if __name__ == "__main__":
    main()