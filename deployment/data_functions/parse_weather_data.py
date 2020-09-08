import json
import math
import pandas as pd
import datetime
import config


def parse_data(today: str, column_names: list) -> 'DataFrame':
    '''Takes list of JSON dicts and target column names, returns
    dataframe containing values of weather variables by day and
    lat lon bin'''

    # empty list to hold results
    rows = []

    # Two things to be done here. First we need to parse the future 7 days
    # of weather prediction data. Then we need to parse the data from the
    # past 5 days and combine the two.
    #
    # future data first

    # construct filename for weather prediction data
    prediction_data_filename = f'{config.RAW_DATA_DIR}future_weather/{today}.json'

    # load future weather data
    with open(prediction_data_filename) as prediction_data_file:
        prediction_data = json.load(prediction_data_file)

    # loop on list of JSON objects - each one corresponts to
    # weather data for a lat, lon bin.

    for geospatial_bin in prediction_data:
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

    # Now for the past data - in this case we have individual
    # files for each day so we need to make a date range
    # and loop over it. We start with yesterday and go
    # back in time 5 days.

    date_yesterday = datetime.datetime.strptime(
        today, '%Y-%m-%d') - datetime.timedelta(5)

    datelist = pd.date_range(date_yesterday, periods=5).tolist()

    for date in datelist:
        date = date.strftime('%Y-%m-%d')

        # construct filename for weather prediction data
        past_data_filename = f'{config.RAW_DATA_DIR}past_weather/{date}.json'

        # load future weather data
        with open(past_data_filename) as past_data_file:
            past_data = json.load(past_data_file)

            for geospatial_bin in past_data:
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
    # seems to have trouble with the pandas datetime when
    # saving the dataframe to parquet so we convert the
    # date to string
    rows_df['date'] = rows_df['date'].dt.date.astype(str)
    rows_df['month'] = pd.DatetimeIndex(rows_df['date']).month.astype('int32')

    # return result as dataframe
    return rows_df
