import json
import math
import pandas as pd
import datetime


def parse_data(data: list, column_names: list) -> 'DataFrame':
    '''Takes list of JSON dicts and target column names, returns
    dataframe containing values of weather variables by day and
    lat lon bin'''

    # empty list to hold records
    rows = []

    # loop on list of JSON objects - each one corresponts to
    # weather data for a lat, lon bin. Two types of list element
    # are present - one for 7 days of future data and one which contains
    # hourly data for a day in the past. We will need to parse each
    # differently
    for geospatial_bin in data:

        # if daily is in the kes for the current bin, the bin is a
        # forecast record containing 7 days worth of future weather data
        # for a lat lon bin
        if 'daily' in geospatial_bin.keys():
            # get lat and lon location of bin
            lat = geospatial_bin['lat']
            lon = geospatial_bin['lon']

            # get list of daily weather data from bin dict
            geospatial_bin_data = geospatial_bin['daily']

            # loop over day list to extract data
            for day in geospatial_bin_data:
                # get date for current day
                date = day['dt']

                # get temperatures for each time of day
                # and average
                day_temp = day['temp']['day']
                evening_temp = day['temp']['eve']
                night_temp = day['temp']['night']
                morning_temp = day['temp']['morn']
                temp = (day_temp + evening_temp +
                        night_temp + morning_temp) / 4

                # Get pressure, humidity and dew point. Note: NOAA pressure data
                # used for training is in units of Pa while openweather uses hPa
                pressure = day['pressure'] * 100
                humidity = day['humidity']
                dew_point = day['dew_point']

                # get wind speed and direction
                wind_speed = day['wind_speed']
                wind_direction = day['wind_deg']

                # decompose wind speed and direction into vector components
                # to match format of NOAA training data
                uwind = -wind_speed * math.sin(math.radians(wind_direction))
                vwind = -wind_speed * math.cos(math.radians(wind_direction))

                # get cloud cover
                cloud_cover = day['clouds']

                # get precipitation - if no precipitation was recorded
                # for a given day, this key will be missing, therfore
                # assign a zero
                rain = day.get('rain', 0.0)

                # Assemble data into list
                row = [
                    date,
                    lat,
                    lon,
                    temp,
                    rain,
                    humidity,
                    dew_point,
                    pressure,
                    uwind,
                    vwind,
                    cloud_cover
                ]

                # add data to growing list of rows
                rows.append(row)

        # if hourly exists in keys, we are looking at a past weather
        # record corresponding to hourly data for one day
        elif 'hourly' in geospatial_bin.keys():
            # get location
            lat = geospatial_bin['lat']
            lon = geospatial_bin['lon']

            # get date from first hour record
            date = geospatial_bin['hourly'][0]['dt']

            # get list of hourly data for bin
            geospatial_bin_data = geospatial_bin['hourly']

            # empty lists to hold hourly data
            temp_list = []
            pressure_list = []
            humidity_list = []
            dew_point_list = []
            wind_speed_list = []
            wind_direction_list = []
            cloud_cover_list = []
            rain_list = []

            # loop on hours in bin to extract data to lists
            for hour in geospatial_bin_data:
                # add weather variables of intrest to respective lists
                temp_list.append(hour['temp'])
                pressure_list.append(hour['pressure'])
                humidity_list.append(hour['humidity'])
                dew_point_list.append(hour['humidity'])
                wind_speed_list.append(hour['wind_speed'])
                wind_direction_list.append(hour['wind_deg'])
                cloud_cover_list.append(hour['clouds'])

                # note: if no precipitation was recorded
                # for a given day, this key will be missing, therfore
                # assign a zero
                rain_list.append(day.get('rain', 0.0))

            # average weather variables over the day
            temp = sum(temp_list) / len(temp_list)
            pressure = sum(pressure_list) / len(pressure_list)
            humidity = sum(humidity_list) / len(humidity_list)
            dew_point = sum(dew_point_list) / len(dew_point_list)
            wind_speed = sum(wind_speed_list) / len(wind_speed_list)
            cloud_cover = sum(cloud_cover_list) / len(cloud_cover_list)
            wind_direction = sum(wind_direction_list) / \
                len(wind_direction_list)

            # decompose wind speed and direction into vector components
            # to match format of NOAA training data
            uwind = -wind_speed * math.sin(math.radians(wind_direction))
            vwind = -wind_speed * math.cos(math.radians(wind_direction))

            # sum rain to get total rainfall for the day
            rain = sum(rain_list)

            # Assemble data into list
            row = [
                date,
                lat,
                lon,
                temp,
                rain,
                humidity,
                dew_point,
                pressure,
                uwind,
                vwind,
                cloud_cover
            ]

            # add data to growing list of rows
            rows.append(row)

    # assemble rows into dataframe with named columns
    rows_df = pd.DataFrame(rows, columns=column_names)

    # set dtype of date & format
    rows_df['date'] = pd.to_datetime(
        rows_df['date'], unit='s')

    # Note our time resolution is at the level of days,
    # so we don't want the time in our date column pyarrow
    # seems to have trouble with when saving the dataframe
    # to parquet so we convert the date to string
    rows_df['date'] = rows_df['date'].dt.date.astype(str)

    # return result as dataframe
    return rows_df
