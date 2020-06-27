import json
import requests
import time
from datetime import datetime, timedelta


def get_current_fires(date_today: str) -> list:
    '''returns active fires as list of 
    JSON dicts.'''

    # empyt list to hold incomming data
    responses = []
    #date_today = datetime.today().strftime('%Y-%m-%d')
    print(f'Will get active fires for: {date_today}')

    # get active fires
    url = f'https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Active_Fires/FeatureServer/0/query?outFields=*&where=1%3D1&f=geojson'
    try:
        response = requests.get(url)

    except requests.exceptions.RequestException as error:
        raise SystemExit(error)

    # get response as string and load into JSON object, append to response list
    text = response.text
    data = json.loads(text)
    responses.append(data)

    # return list of JSON objects
    return responses


def get_weather_forecast(key: str, lat_lon_bins: list, date_today: str) -> list:
    '''Takes list of lat, lon bins and API key. Returns 
    corresponding 7 day weather forecast data as list of 
    JSON dicts.'''

    # empyt list to hold incomming data
    responses = []
    #date_today = datetime.today().strftime('%Y-%m-%d')
    print(f'Will get 7 day weather forecast data starting: {date_today}')

    # loop on lat lon bins to get data
    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        # get 7 days of prediction data - API allows this all in one call
        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly&appid={key}'

        try:
            response = requests.get(url)

        except requests.exceptions.RequestException as error:
            raise SystemExit(error)

        # get response as string and load into JSON object, append to response list
        print(f'Got 7 day forecast for: {date_today} - {lat}, {lon}')
        text = response.text
        data = json.loads(text)
        responses.append(data)

        # wait, so we don't hit the API server too hard
        time.sleep(1)

    # return list of JSON objects
    return responses


def get_past_weather(key: str, lat_lon_bins: list, date_yesterday: str) -> list:
    '''Takes list of lat, lon bins and API key. Returns 
    yesterday's weather data as list of JSON dicts.'''

    # Construct target date
    responses = []
    #date_yesterday = (datetime.today() - timedelta(1)).strftime('%Y-%m-%d')
    timestamp_yesterday = int(time.mktime(
        datetime.strptime(date_yesterday, '%Y-%m-%d').timetuple()))

    print(f'Will get weather data for: {date_yesterday}')
    print(f'Timestamp reads: {timestamp_yesterday}')

    # loop on lat lon bins to get data
    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp_yesterday}&appid={key}'

        try:
            response = requests.get(url)

        except requests.exceptions.RequestException as error:
            raise SystemExit(error)

        # get response as string and load into JSON object, append to response list
        print(f'Got past data for: {lat}, {lon} on {date_yesterday}')
        text = response.text
        data = json.loads(text)
        responses.append(data)

        # wait, so we don't hit the API server too hard
        time.sleep(1)

    # return list of JSON objects
    return responses
