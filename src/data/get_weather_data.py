import json
import requests
import time
import datetime


def get_data(key: str, lat_lon_bins: list) -> list:

    responses = []
    days = 5
    date_today = datetime.datetime.today()
    date_list = [int((date_today - datetime.timedelta(days=x)).timestamp())
                 for x in range(days)]

    for lat_lon_bin in lat_lon_bins:
        lat = lat_lon_bin[0]
        lon = lat_lon_bin[1]

        url = f'https://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&exclude=current,minutely,hourly&appid={key}'

        try:
            response = requests.get(url)

        except requests.exceptions.RequestException as error:
            raise SystemExit(error)

        text = response.text
        data = json.loads(text)
        responses.append(data)

        time.sleep(1)

        for date in date_list:
            url = f'https://api.openweathermap.org/data/2.5/onecall/timemachine?lat={lat}&lon={lon}&dt={date}&appid={key}'

            try:
                response = requests.get(url)

            except requests.exceptions.RequestException as error:
                raise SystemExit(error)

            text = response.text
            data = json.loads(text)
            responses.append(data)

            time.sleep(1)

    return responses
