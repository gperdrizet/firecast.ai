'''Converts raw NetCDF files from NOAA into parquet. Cleans
up extraneous collumns and missing values. Also does a first pass
coarse geospatial filter to reduce the size of the data set using a lat, lon
bounding box arround the continental United States.
'''

import logging
import os.path
from multiprocessing import Pool
import xarray as xr
import config

logging.basicConfig(
    filename=config.CONVERT_CLEAN_RAW_NOAA_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

def netcdf_to_data(filename):
    '''Takes name of netCDF file, uses xarray to read
    file into an xarray data set, then converts to
    pandas dataframe and returns.'''

    dataset = xr.open_dataset(filename)
    data = dataset.to_dataframe()

    return data

def clean_noaa_narr_data(data, data_type):
    '''Takes raw NOAA NARR weather data frame, removes
    unnecessary colums & index levels. Leaves lat, lon
    and value. Renames value column after data_type
    Leaves index as datetime of observation.'''

    data.index = data.index.droplevel([1, 2])
    data.rename(columns={data.columns[3]:data_type}, inplace=True)
    data.drop(['Lambert_Conformal'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)

    return data

def spatial_filter_coarse(data):
    '''Takes dataframe containing lat, lon columns. Returns
    only rows which fall inside California bounding box
    coordinates.'''

    data = data.loc[(data['lat'] >= config.LAT_START) &
                    (data['lat'] <= config.LAT_END)]

    data = data.loc[(data['lon'] >= config.LON_START) &
                    (data['lon'] <= config.LON_END)]

    return data


def convert_clean_noaa_weather_data(data_year):
    '''Reads NOAA data from orignal netCDF file.
    Cleans up columns and writes to parquet'''
    
    # loop on weather variable
    for variable, data_type in config.DATA_TYPES.items():

        logging.info(' Parsing data from %s', data_year)

        # construct input and output file names
        input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                      variable + "." + str(data_year) + "." +
                      config.NOAA_NARR_FILE_EXT)

        output_file = (config.WEATHER_DATA_BASE_PATH + config.PARSED_DATAFILE_SUBDIR +
                       variable + "." + str(data_year) + ".parquet")

        # if we already have the data, skip 
        if os.path.isfile(output_file):
            logging.info(' %s data for %s exits, skipping parse', data_type, data_year)

        # if we don't have the data format, clean and filter the data
        else:
            logging.info(' Reading from %s', input_file)
            logging.info(' Writing to %s', output_file)

            data = netcdf_to_data(input_file)
            data = clean_noaa_narr_data(data, variable)
            data = spatial_filter_coarse(data)
            data.reset_index(inplace=True)

            logging.info(' Dataframe has columns: %s', list(data))
            logging.info(' Setting %s column to type %s', variable, data_type)

            data = data.astype({'lat': 'float32', 'lon': 'float32', variable: 'float32'})

            logging.info(data.dtypes)
    
            # output to parquet
            data.to_parquet(output_file)

def main():
    '''Paralelize opperations over years on local machine'''

    logging.info(' DATA_TYPES contains %s', config.DATA_TYPES)

    # Create processing pool and assign each member a data year to work on
    with Pool(config.CONVERT_CLEAN_PROCESSES) as pool:
        pool.map(convert_clean_noaa_weather_data, config.DATA_YEARS)

    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
