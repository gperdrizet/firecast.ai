from flask import Flask
from flask import request
from scipy import spatial
import pandas as pd
import numpy as np
import json

app = Flask(__name__)

input_file = '../data/predictions/2020-06-03_formatted_predictions.parquet'
predictions = pd.read_parquet(input_file)

lat_lon_bins = predictions[['lat', 'lon']].astype(float).to_numpy()
predictions['day'] = pd.to_datetime(predictions['day'])
predictions['lat'] = predictions['lat'].astype(float)
predictions['lon'] = predictions['lon'].astype(float)
print(predictions)


@app.route('/fire_risk_by_location')
def fire_risk_by_location(methods=['GET']):
    # http://72.132.251:9998:9998/fire_risk_by_location?lat=34.067&lon=-114.6695
    lat = float(request.args.get('lat'))
    lon = float(request.args.get('lon'))

    query_coords = [lat, lon]
    distance, index = spatial.KDTree(lat_lon_bins).query(query_coords)
    bin_lat = lat_lon_bins[index][0]

    result = predictions[(predictions['lat'] == bin_lat)
                         & (predictions['lon'] == bin_lon)]

    result = result.to_json(orient='records')

    return result


@app.route('/fire_risk_by_date')
def fire_risk_by_date(methods=['GET']):
    # http://72.132.251:9998/fire_risk_by_date?start=2020-06-06&end=2020-06-07
    start_date = pd.to_datetime(request.args.get('start'))
    end_date = pd.to_datetime(request.args.get('end'))

    result = predictions[(predictions['day'] >= start_date)
                         & (predictions['day'] <= end_date)]

    result = result.to_json(orient='records')

    return result


if __name__ == '__main__':
    app.run(host='192.168.1.238', port=9998, debug=True)
