# California Wildfire Probability Prediction

The goal of this project is to build a machine learning model which can predict wildfire ignition risk in California from publicly available meteorology and fire activity data.

#### -- Project Status: [Active]

## Introduction

Wild fires are common, destructive and deadly natural disasters. Current meteorology based risk prediction methods can be improved upon by: 

1. Applying modern machine learning techniques
2. Leveraging historical wildfire data for model training and validation

## Project Description

This project is currently in the data wrangling phase. The goal is to collect and regularize the following data types in space and time:

* Locations of past fire ignition sites
* Past fire sizes
* Past fire severity codes
* Surface Air Temperature
* Air Temperature at 2 Meters
* Accumulated Precipitation
* Relative Humidity at 2 Meters
* Dew Point Temperature
* Surface Pressure
* East-west Component of Wind
* North-south Component of Wind
* Vegitation Coverage',
* Downward Long Wave EM Flux
* Downward Short Wave EM Flux
* Low Height Cloud Coverage
* High Height Cloud Coverage
* Medium Height Cloud Coverage
* Planetary Boundary Layer Height
* Precipitation Rate
* Visibility
* Upward Long Wave EM Flux

This historical dataset will be used to train a machine learning model to predict wild fire ignition probability. After initial training and validation is complete the next phase of the project will be to deploy the model as a real time prediction service utilizing current fire activity data and weather predictions.

See the full [project proposal](https://github.com/gperdrizet/wildfire/tree/master/docs/project_proposal.md) for more info.

## Data Sources
1. Historical wildfire activity: United States Department of Agriculture Research Data Archive, [*Spatial wildfire occurrence data for the United States, 1992-2015*](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.4)<sup>1</sup>
2. Historical metrology data: National Oceanic and Atmospheric Administration, [*North American Regional Reanalysis*](https://catalog.data.gov/dataset/ncep-north-american-regional-reanalysis-narr)<sup>2</sup>

## Featured Notebooks

* [Training data exploratory analysis](https://github.com/gperdrizet/wildfire/tree/master/notebooks/training_data_exploration.ipynb)

### Methods Used

* Inferential Statistics
* Machine Learning
* Data Visualization
* Cartographic Projection
* Time Series Analysis
* Feature Engineering
* Predictive Modeling

### Technologies

* Python
* PySpark
* Luigi
* Flask
* Tensorflow
* Keras
* Scikit-Learn
* Pandas
* NumPy
* Shaply
* GeoPandas
* Xarray
* Matplotlib
* Seaborn

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Training data sets are being kept [here](https://github.com/gperdrizet/wildfire/tree/master/data/training_data/) within this repo.
3. Data exploration/transformation notebooks are being kept [here](https://github.com/gperdrizet/wildfire/tree/master/notebooks)
4. Finalized scripts for data aquisition and transformation are being kept [Here](https://github.com/gperdrizet/wildfire/tree/master/python)  


## Contributing Members

**Team Lead (Contact) : [George Perdrizet](https://github.com/gperdrizet)**

## References
1. Short, Karen C. 2017. *Spatial wildfire occurrence data for the United States, 1992-2015* [FPA_FOD_20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
2. NCEP Reanalysis data provided by the NOAA/OAR/ESRL PSD, Boulder, Colorado, USA, from their Web site at https://www.esrl.noaa.gov/psd/