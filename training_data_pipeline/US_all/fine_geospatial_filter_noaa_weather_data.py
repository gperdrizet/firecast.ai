''' Takes list of 'keeper' lat, lon points which are within the boarders
of the continental US, uses them to filter the weather data and discard
points outside of the US
'''

import os.path
import pandas as pd
import numpy as np
from multiprocessing import Pool
import logging
import config

logging.basicConfig(
    filename=config.FINE_GEOSPATIAL_FILTER_NOAA_WEATHER_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

KEEPER_GEOSPATIAL_BINS = pd.read_parquet(config.TARGET_GEOSPATIAL_BINS_FILE)

def spatial_filter_fine(data):
    '''Takes dataframe and does innerjoin with US geospatial bins'''
    keepers = data.merge(KEEPER_GEOSPATIAL_BINS, left_on=['lat', 'lon'],
                         right_on=['lat', 'lon'], how='inner')

    return keepers

def parallelize(function, data, n_processes, jobs_per_process):
    '''Parallelizes a function. Takes function name, dataframe
    and number of threads. Splits up function call over
    avalible threads. Joins and returns the results.'''
    
    # split data into chunks
    data_split = np.array_split(data, (n_processes * jobs_per_process))
    
    # create multiprocessing pool
    with Pool(n_processes) as pool:
        
        # give each worker a chunk of data and a function to pass the data to
        result = pd.concat(pool.imap(function, data_split))

    pool.close()
    
    # join and return results
    pool.join()
    return result

def main():
    '''Loops over weather data years and variables to do fine geospatial filtering'''
    for data_year in config.DATA_YEARS:
        for variable, data_type in config.DATA_TYPES.items():


            # construct IO file names
            input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                          variable + "." + str(data_year) + ".parquet")

            output_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                           variable + "." + str(data_year) + ".us_only.parquet")

            # if we already have the data, skip it. If not, go ahead and run the filter
            if os.path.isfile(output_file):
                logging.info(' %s data for %s exits, skipping parse', data_type, data_year)

            else:
                logging.info(' Fine geospatialy filtering %s', input_file)

                # read in data from disk
                data = pd.read_parquet(input_file)

                # parallelize spatial filtering of data on local machine
                data = parallelize(spatial_filter_fine, data, 
                                   config.FINE_SPATIAL_FILTER_PROCESSES, config.JOBS_PER_PROCESS).dropna()

                # write result to disk
                data.to_parquet(output_file)
                
if __name__ == "__main__":
    main()