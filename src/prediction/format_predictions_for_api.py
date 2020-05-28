import pandas as pd
from datetime import datetime


def format_for_api(data: 'numpy.ndarray', lat_lon_bins: list) -> 'DataFrame':
    lat_lon_bins = pd.DataFrame(lat_lon_bins, columns=['lat', 'lon'])
    date_list = pd.date_range(datetime.today(), periods=7).tolist()

    i = 0
    days = []
    for day in data:
        ignition_risk = day.reshape((410, 1))
        ignition_risk = pd.DataFrame(data=ignition_risk.flatten(),
                                     columns=['ignition_risk'])
        ignition_risk['day'] = date_list[i].date()
        ignition_risk = pd.concat([ignition_risk, lat_lon_bins], axis=1)

        days.append(ignition_risk)

        i += 1

    predictions = pd.concat(days)

    return predictions
