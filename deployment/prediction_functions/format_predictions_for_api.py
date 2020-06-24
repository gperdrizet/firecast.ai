import pandas as pd
from datetime import datetime


def format_for_api(data: 'numpy.ndarray', lat_lon_bins: list) -> 'DataFrame':
    '''Takes a 3D numpy array of predictions by day and location with a list of lat lon bins
    returns a formatted dataframe containing predictions by day and location'''

    # Load orignal lat lon bins
    lat_lon_bins = pd.DataFrame(lat_lon_bins, columns=['lat', 'lon'])

    # construct prediction date list (7 future days from today)
    date_list = pd.date_range(datetime.today(), periods=7).tolist()

    # loop on the first dimension of the numpy prediction array (days)
    i = 0
    days = []
    for day in data:
        # 410 is the number of spatial bin, reshape from wide to long
        # so we can make the risk prediction value a column in a dataframe
        ignition_risk = day.reshape((410, 1))
        ignition_risk = pd.DataFrame(data=ignition_risk.flatten(),
                                     columns=['ignition_risk'])

        # add day column
        ignition_risk['day'] = date_list[i].date()

        # add lat, lon bins
        ignition_risk = pd.concat([ignition_risk, lat_lon_bins], axis=1)

        # add day worth of prediction data we just constructed to
        # growing master list
        days.append(ignition_risk)

        i += 1

    # concatenate over days
    predictions = pd.concat(days)

    # set dtype of date & format
    predictions['day'] = pd.to_datetime(
        predictions['day'])

    # Note our time resolution is at the level of days,
    # so we don't want the time in our date column pyarrow
    # seems to have trouble with when saving the dataframe
    # to parquet so we convert the date to string
    predictions['day'] = predictions['day'].dt.date.astype(str)

    # return resulting dataframe
    return predictions
