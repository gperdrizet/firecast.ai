'''Collection of helper functions for use in parsing and
formatting NOAA NARR weather data'''

import numpy as np
import pandas as pd
import xarray as xr

from multiprocessing import Pool

import CA_only.config as config
from CA_only.spatial_data_helper_functions import spatial_filter_coarse
from CA_only.spatial_data_helper_functions import spatial_filter_fine


def netcdf_to_data(filename: str) -> 'DataFrame':
    '''Takes name of netCDF file, uses xarray to read
    file into an xarray data set, then converts to
    pandas dataframe and returns.'''

    dataset = xr.open_dataset(filename)
    data = dataset.to_dataframe()

    return data


def clean_noaa_narr_data(data: 'DataFrame', data_type: str) -> 'DataFrame':
    '''Takes raw NOAA NARR weather data frame, removes
    unnecessary colums & index levels. Leaves lat, lon
    and value. Renames value column after data_type
    Leaves index as datetime of observation.'''

    data.index = data.index.droplevel([1, 2])
    data.rename(columns={data.columns[3]: data_type}, inplace=True)
    data.drop(['Lambert_Conformal'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)
    data.columns = data.columns.str.replace(".", "_")

    return data


def first_pass_parse_noaa_weather_data(data_year: int) -> 'DataFrame':
    '''Reads NOAA data from orignal netCDF file.
    Cleans up columns and writes to parquet. Also uses a
    lat, lon bounding box around California to discard points and 
    reduce the size of the dataset'''

    # loop on weather variable
    for data_type in config.DATA_TYPES.keys():

        # construct input and output file names
        input_file = (config.WEATHER_DATA_BASE_PATH + config.COMPLETE_RAW_DATAFILE_SUBDIR +
                      data_type + "." + str(data_year) + '.nc')

        output_file = (config.WEATHER_DATA_BASE_PATH + config.TEMP_PARSED_DATAFILE_SUBDIR +
                       data_type + "." + str(data_year) + ".california_box.parquet")

        # read netcdf file with xarray
        data = netcdf_to_data(input_file)

        # remove unncesasry columns, rename data columns and set date time index
        data = clean_noaa_narr_data(data, data_type)

        # discard data outside of bounding box encompassing california
        data = spatial_filter_coarse(data, config.BOUNDING_BOX)
        data.reset_index(inplace=True)

        # output to parquet
        data.to_parquet(output_file)

        print("Finished: "+output_file)


def second_pass_parse_noaa_weather_data(data_year):
    '''Further reduces size of the dataset by keeping
    only points within California boarders'''

    # loop on weather variable
    for data_type in config.DATA_TYPES.keys():

        # construct input and output file names
        input_file = (config.WEATHER_DATA_BASE_PATH + config.TEMP_PARSED_DATAFILE_SUBDIR +
                      data_type + "." + str(data_year) + ".california_box.parquet")

        output_file = (config.WEATHER_DATA_BASE_PATH + config.TEMP_PARSED_DATAFILE_SUBDIR +
                       data_type + "." + str(data_year) + ".california_only.parquet")

        # read data which has been preprocessed with coarse bounding box geospatial filter
        data = pd.read_parquet(input_file)

        # use shaply and California polygon to discard data outside of California's boarders
        data = spatial_filter_fine(data)

        # output to parquet
        data.to_parquet(output_file)

        print("Finished: "+output_file)


def collect_by_year(data_year):
    '''Takes a year and dict of weather variable data types, returns single
    dataframe containing all weather data for that year'''

    # get list of datatypes
    data_types = list(config.DATA_TYPES.keys())

    # read first dataset into dataframe so we have something to join with
    first_variable = data_types[0]

    first_input_file = (config.WEATHER_DATA_BASE_PATH + config.COMPLETE_PARSED_DATAFILE_SUBDIR +
                        first_variable + "." + str(data_year) + ".california_only.parquet")

    data = pd.read_parquet(first_input_file)

    # start loop on second data_type
    for data_type in data_types[1:]:

        # read new datafile
        input_file = (config.WEATHER_DATA_BASE_PATH + config.COMPLETE_PARSED_DATAFILE_SUBDIR +
                      data_type + "." + str(data_year) + ".california_only.parquet")

        incomming_data = pd.read_parquet(input_file)

        # make sure column name formatting matches
        incomming_data.columns = incomming_data.columns.str.replace("[.]", "_")

        # merge new datafile with master
        data = pd.merge(data, incomming_data, on=[
                        'lat', 'lon', 'time'], how='outer')

    # write result
    output_file = (config.WEATHER_DATA_BASE_PATH + config.COMPLETE_PARSED_DATAFILE_SUBDIR +
                   "all." + str(data_year) + ".california_only.parquet")

    data.to_parquet(output_file)


def parallelize(function, data: 'DataFrame', n_processes: int, jobs_per_process: int) -> 'DataFrame':
    '''Parallelizes a function. Takes function name, dataframe
    and number of threads. Splits up function call over
    avalible threads. Joins and returns the results.'''
    data_split = np.array_split(data, (n_processes * jobs_per_process))
    with Pool(n_processes) as pool:
        result = pd.concat(pool.imap(function, data_split), sort=True)

    pool.close()
    pool.join()

    return result
