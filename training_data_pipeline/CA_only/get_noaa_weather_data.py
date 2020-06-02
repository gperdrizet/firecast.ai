'''Downloads data for each weather variable in year range specified by config.py from NOAA.gov'''
import os
import requests


def get_weather_data(base_url: str, data_years: list, data_types: list, output_path: str) -> bool:
    '''Loops on years and data types, checks to see if we already have the
    data. If not, downloads from NOAA.gov. Saves file in source format'''

    for data_year in data_years:
        for data_type in data_types:

            output_file = (output_path +
                           data_type + "." + str(data_year) + ".nc")

            # Check to see if we already have the datafile
            if os.path.isfile(output_file):
                continue

            # If we don't already have the datafile, download it
            else:
                # Format download url and output file name
                url = (base_url + data_type + '.' + str(data_year) + '.nc')

                try:
                    response = requests.get(url)

                except:
                    continue

                else:
                    print(f'Writing: {output_file}')
                    open(output_file, 'wb').write(response.content)

    return True
