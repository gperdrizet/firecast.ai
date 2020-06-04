'''Collects all years of each weather data type into a single csv file'''

import logging
import os.path
import pandas as pd
from multiprocessing import Pool
import config

logging.basicConfig(
    filename=config.COLLECT_NOAA_WEATHER_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)


def collect(data_year):
    '''Takes year as input, loops over weather variable types for that year
    and collects each into a single dataframe. Outputs data to parquet
    '''
    
    # construct output file name
    output_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR + str(data_year) + "_us_only_all.parquet")

    # check to see if we alread have the data, if not go ahead and make it
    if os.path.isfile(output_file):
        logging.info(' Data for %s exits, skipping parse', data_year)
        
    else:  
        # read first dataset into dataframe so we have something to join with
        
        # get first weather variable name
        variables = list(config.DATA_TYPES)
        first_variable = variables[0]

        logging.info(' %s, %s', data_year, first_variable)

        # construct input file name
        input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                      first_variable + "." + str(data_year) + ".us_only.parquet")

        logging.info(' Input: %s', input_file)

        # read first weather dataset
        data = pd.read_parquet(input_file)
        
        # replace "." with "-" in column names. This was found to cause problems later on
        data.columns = data.columns.str.replace("[.]", "_")

        # start loop on second data_type, joining sequentialy to first
        for variable in variables[1:]:

            logging.info(' %s, %s', data_year, variable)
            
            # construct input filename
            input_file = (config.WEATHER_DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                          variable + "." + str(data_year) + ".us_only.parquet")

            logging.info(' Input: %s', input_file)

            # read data
            incomming_data = pd.read_parquet(input_file)
            
            # replace "." with "-" in column names. This was found to cause problems later on
            incomming_data.columns = incomming_data.columns.str.replace("[.]", "_")
            
            # merge the new data we just read with the growing master dataframe for the year
            data = pd.merge(data, incomming_data, on=['lat', 'lon', 'time'], how='outer')

        logging.info(' Output: %s', output_file)

        logging.info(output_file)
        
        # write combined data for the year to parquet
        data.to_parquet(output_file)

if __name__ == "__main__":
    '''Paralelize opperations over years on local machine'''
    
    # creates 
    with Pool(config.COLLECT_PROCESSES) as pool:
        
        # Create processing pool and assign each member a data year to work on
        pool.map(collect, config.DATA_YEARS)

    pool.close()
    pool.join()
