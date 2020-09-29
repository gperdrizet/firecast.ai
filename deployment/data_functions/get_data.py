import json
import requests
import time
import config
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

    # list containing the names of the variables
    # we expect to recive from the weather API
    expected_vars = config.WEATHER_API_VARIABLE_NAMES

    # empty list to hold incomming data
    responses = []
    print(f'Will get 7 day weather forecast data starting: {date_today}')

    # loop on lat lon bins to get data
    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        # get 7 days of prediction data - API allows this all in one call
        # construct API call
        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly&appid={key}'

        # set response to empty string and attempt count to 0
        data = ''
        attempts = 0

        # attempt to get the data a max of 3 times until sucessfull
        while (attempts < 3) & (set(expected_vars).issubset(data) != True):
            try:
                # query API
                response = requests.get(url)

            except requests.exceptions.RequestException as error:
                raise SystemExit(error)

            # extract text from response
            text = response.text

            # get json object from text string
            data = json.loads(text)

            attempts += 1

        # if we tried three times and still did not get the data we need
        # print a warning before giving up and moving on
        if set(expected_vars).issubset(data):
            print(f'Could not get data for: {lat}, {lon}')
            print(f'Response reads: {text}')

        elif set(expected_vars).issubset(data) != True:
            # If API call was sucessfull, append to response list
            print(f'Got 7 day forecast for: {date_today} - {lat}, {lon}')
            responses.append(data)

        # wait, so we don't hit the API server too hard
        time.sleep(1)

    # return list of JSON objects
    return responses


def get_past_weather(key: str, lat_lon_bins: list, date_yesterday: str) -> list:
    '''Takes list of lat, lon bins and API key. Returns 
    yesterday's weather data as list of JSON dicts.'''

    # list containing the names of the variables
    # we expect to recive from the weather API
    expected_vars = ['lat', 'lon']

    # Construct target date
    responses = []
    timestamp_yesterday = int(time.mktime(
        datetime.strptime(date_yesterday, '%Y-%m-%d').timetuple()))

    print(f'Will get weather data for: {date_yesterday}')

    # loop on lat lon bins to get data
    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        # get 1 past day of prediction data - API only allows one day per call
        # construct API call
        url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={timestamp_yesterday}&appid={key}'

        # set response to empty string and attempt count to 0
        data = ''
        attempts = 0

        # attempt to get the data a max of 3 times until sucessfull
        while (attempts < 3) & (set(expected_vars).issubset(data) != True):
            try:
                # query API
                response = requests.get(url)

            except requests.exceptions.RequestException as error:
                raise SystemExit(error)

            # extract text from response
            text = response.text

            # get json object from text string
            data = json.loads(text)

            attempts += 1

        # if we tried three times and still did not get the data we need
        # print a warning before giving up and moving on
        if set(expected_vars).issubset(data) != True:
            print(f'Could not get data for: {lat}, {lon}')
            print(f'Response reads: {text}')

        elif set(expected_vars).issubset(data):
            # get response as string and load into JSON object, append to response list
            print(f'Got past data for: {lat}, {lon} on {date_yesterday}')
            responses.append(data)

        # wait, so we don't hit the API server too hard
        time.sleep(1)

    # return list of JSON objects
    return responses
