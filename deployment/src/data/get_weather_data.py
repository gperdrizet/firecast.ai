import json
import requests
import time
import datetime


def get_data(key: str, lat_lon_bins: list) -> list:
    '''Takes list of lat, lon bins and API key. Returns list
    of JSON dicts.

    Time span is 7 days future and 5 days past from current date.'''

    # Construct list of target dates
    responses = []
    days = 5
    date_today = datetime.datetime.today()
    date_list = [int((date_today - datetime.timedelta(days=x)).timestamp())
                 for x in range(days)]

    print(f'Will get weather forecast data for: {date_list}')

    # loop on lat lon bins to get data
    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        # first, get 7 days of prediction data - API allows this all in one call
        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly&appid={key}'

        try:
            response = requests.get(url)

        except requests.exceptions.RequestException as error:
            raise SystemExit(error)

        # get response as string and load into JSON object, append to response list
        print(f'Got forecast for: {lat}, {lon}')
        text = response.text
        data = json.loads(text)
        responses.append(data)

        # wait, so we don't hit the API server too hard
        time.sleep(1)

        # loop on target dates to get past weather data - API does not allow one
        # call or a date range here, must get 24 hrs of data by day
        for date in date_list:
            url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={date}&appid={key}'

            try:
                response = requests.get(url)

            except requests.exceptions.RequestException as error:
                raise SystemExit(error)

            # get response as string and load into JSON object, append to response list
            print(f'Got past data for: {lat}, {lon} on {date}')
            text = response.text
            data = json.loads(text)
            responses.append(data)

            # wait, so we don't hit the API server too hard
            time.sleep(1)

    # return list of JSON objects
    return responses
