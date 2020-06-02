'''First pass weather data parse: read netcdf into xarray dataset
and covert to pandas dataframe, clean up extraneous columns and
missing data. Do coarse geospatial filter with lat, lon bounding 
box around California'''

import warnings
from multiprocessing import Pool
import pandas as pd
import CA_only.config as config
from CA_only.weather_data_helper_functions import first_pass_parse_noaa_weather_data
from CA_only.weather_data_helper_functions import second_pass_parse_noaa_weather_data
from CA_only.weather_data_helper_functions import parallelize
from CA_only.spatial_data_helper_functions import keep_ca_points


def parse_weather_data(data_years: list) -> bool:
    # NOTE: the orignal netcdf files from NOAA use an arbitrarily large value to mask NANs
    # xarray correctly identifys and decodes these values to NAN, but it complains
    # about it the whole time...
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Create processing pool and assign each member a data year to work on
        with Pool(config.FIRST_PASS_PARSE_PROCESSES) as pool:
            pool.map(first_pass_parse_noaa_weather_data, data_years)

        pool.close()
        pool.join()

    '''Use geopandas and shaply to get the set of lat lon points from the
    weather dataset which are actualy in California. This opperation is 
    computationaly expensive, only do it once on a sample weather data
    file. Once we have our "keeper" bins, we can use joins to filter the 
    whole data set.'''

    # pick and load one data file
    data_year = 1992
    data_type = 'air.sfc'
    input_file = (config.WEATHER_DATA_BASE_PATH + config.TEMP_PARSED_DATAFILE_SUBDIR +
                  data_type + "." + str(data_year) + ".california_box.parquet")

    data = pd.read_parquet(input_file)

    # get complete set of corrdinates for one timepoint, discard everything else
    data = data[data['time'] == '1992-01-01 00:00:00']
    data.drop(['air_sfc'], axis=1, inplace=True)

    # parallelize the point checking
    KEEPER_GEOSPATIAL_BINS = parallelize(keep_ca_points, data,
                                         config.FINE_SPATIAL_FILTER_PROCESSES,
                                         config.JOBS_PER_PROCESS).dropna()

    KEEPER_GEOSPATIAL_BINS.reset_index(inplace=True, drop=True)
    KEEPER_GEOSPATIAL_BINS.drop_duplicates(inplace=True)
    KEEPER_GEOSPATIAL_BINS.drop('time', axis=1, inplace=True)

    # write to file for later use in parsing wildfire data
    KEEPER_GEOSPATIAL_BINS.to_parquet(config.TARGET_GEOSPATIAL_BINS_FILE)

    '''Second pass - fine geospatial filter to get only points inside of
    California's boarders'''
    with Pool(config.SECOND_PASS_PARSE_PROCESSES) as pool:
        pool.map(second_pass_parse_noaa_weather_data, data_years)

    pool.close()
    pool.join()

    return True
