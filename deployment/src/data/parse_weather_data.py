import json
import math
import pandas as pd
import datetime


def parse_data(data: dict, column_names: list) -> 'DataFrame':
    rows = []

    for geospatial_bin in data:
        if 'daily' in geospatial_bin.keys():
            lat = geospatial_bin['lat']
            lon = geospatial_bin['lon']
            geospatial_bin_data = geospatial_bin['daily']

            for day in geospatial_bin_data:
                date = day['dt']

                day_temp = day['temp']['day']
                evening_temp = day['temp']['eve']
                night_temp = day['temp']['night']
                morning_temp = day['temp']['morn']
                temp = (day_temp + evening_temp +
                        night_temp + morning_temp) / 4

                # NOAA data is in units of Pa while openweather uses hPa
                pressure = day['pressure'] * 100
                humidity = day['humidity']
                dew_point = day['dew_point']

                wind_speed = day['wind_speed']
                wind_direction = day['wind_deg']

                uwind = -wind_speed * math.sin(math.radians(wind_direction))
                vwind = -wind_speed * math.cos(math.radians(wind_direction))

                cloud_cover = day['clouds']
                rain = day.get('rain', 0.0)

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

                rows.append(row)

        elif 'hourly' in geospatial_bin.keys():
            lat = geospatial_bin['lat']
            lon = geospatial_bin['lon']
            date = geospatial_bin['hourly'][0]['dt']
            geospatial_bin_data = geospatial_bin['hourly']

            temp_list = []
            pressure_list = []
            humidity_list = []
            dew_point_list = []
            wind_speed_list = []
            wind_direction_list = []
            cloud_cover_list = []
            rain_list = []

            for hour in geospatial_bin_data:
                temp_list.append(hour['temp'])
                pressure_list.append(hour['pressure'])
                humidity_list.append(hour['humidity'])
                dew_point_list.append(hour['humidity'])
                wind_speed_list.append(hour['wind_speed'])
                wind_direction_list.append(hour['wind_deg'])
                cloud_cover_list.append(hour['clouds'])
                rain_list.append(day.get('rain', 0.0))

            temp = sum(temp_list) / len(temp_list)

            pressure = sum(pressure_list) / len(pressure_list)
            humidity = sum(humidity_list) / len(humidity_list)
            dew_point = sum(dew_point_list) / len(dew_point_list)

            wind_speed = sum(wind_speed_list) / len(wind_speed_list)
            wind_direction = sum(wind_direction_list) / \
                len(wind_direction_list)

            uwind = -wind_speed * math.sin(math.radians(wind_direction))
            vwind = -wind_speed * math.cos(math.radians(wind_direction))

            cloud_cover = sum(cloud_cover_list) / len(cloud_cover_list)
            rain = sum(rain_list)

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

            rows.append(row)

    rows_df = pd.DataFrame(rows, columns=column_names)
    rows_df['date'] = pd.to_datetime(
        rows_df['date'], unit='s')
    rows_df['date'] = rows_df['date'].dt.date

    return rows_df
