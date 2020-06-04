'''Downloads data for each weather variable in year range specified by config.py from NOAA.gov'''

import logging
import os.path
import requests
from requests.exceptions import HTTPError
import config

# set up logging
logging.basicConfig(
    filename=config.GET_WEATHER_DATA_LOG,
    level=logging.DEBUG
)

logging.captureWarnings(True)

def main():
    '''Loops on years and data types, checks to see if we already have the
    data. If not, downloads from NOAA.gov. Saves file in source format'''

    for data_year in config.DATA_YEARS:
        for data_type in config.DATA_TYPES:

            output_file = (config.DATA_BASE_PATH + config.RAW_DATAFILE_SUBDIR +
                           data_type + "." + str(data_year) + "." +
                           config.NOAA_NARR_FILE_EXT)

            # Check to see if we already have the datafile
            if os.path.isfile(output_file):
                logging.info(' %s data for %s exits, skipping download', data_type, data_year)

            # If we don't already have the datafile, download it
            else:
                # Format download url and output file name
                url = (config.NOAA_BASE_URL + data_type + "." + str(data_year) +
                       "." + config.NOAA_NARR_FILE_EXT)
                
                logging.info(' Getting %s data for %s', data_type, data_year)
                logging.info(' URL: %s', url)

                try:
                    response = requests.get(url)

                except HTTPError as http_err:
                    logging.error(' HTTP error occurred: %s', http_err)

                else:
                    open(output_file, 'wb').write(response.content)

if __name__ == "__main__":
    main()
    