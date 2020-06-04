# California Wildfire Risk Prediction: Deployment Plan

![Deployment diagram](https://github.com/gperdrizet/wildfire_production/blob/master/project_info/figures/deployment_diagram.png?raw=true)

The deployment will have two major components: 

1. Luigi pipeline - retrieves weather predictions via API and predicts wildfire risk using the trained pre-trained LSTM neural net.
2. Flask API - makes the predictions publicly available. 
       
The Luigi data pipeline has two main inputs: weather prediction data from the OpenWeatherMap One Call API and trained model weights. The pipeline gets raw weather data, transforms and formats it for input into the neural network, runs the neural network using pre-trained weights and optimized hyperparameters and finally collects and formats the predictions for use by the API. The flask API receives updated predictions from the data pipeline accepts API calls to access the data.

The live prediction data will be stored as parquet files locally on the same server which runs the data pipeline. Because the deployment only predicts fire risk up to 7 days into the future and only needs 5 days of past data to do so, the actual data overhead in deployment is relatively small.

Predictions will be archived so that model performance can be evaluated when wildfire ignition data becomes available.

The major tools needed to build and deploy this system are as follows:

1. Pandas & Numpy for data manipulation
2. Geopandas and Shapely for manipulation of spatial data
3. Xarray for meteorological data
4. Requests for API access
5. Luigi for data pipeline management
6. Flask for API access to prediction data
7. Tensorflow and Keras for prediction


