# California Wildfire Risk Prediction

The goal of this project is to build a machine learning model which can predict wildfire ignition risk in California from publicly available meteorology and fire activity data.

#### -- Project Status: [Active]

## Introduction

Wild fires are common, destructive and deadly natural disasters. Current meteorology based risk prediction methods can be improved upon by: 

1. Applying modern machine learning techniques
2. Leveraging historical wildfire data for model training and validation

## Project Description

This project uses a parallel LSTM neural network to predict geospatialy resolved wildfire ignition risk in California. The model was trained on a combined dataset produced from the USDA historical wildfire activity dataset(1) and meterological data from NOAA's North American Regional Reanalysis(2). This project is currently in the deployment phase. Live prediction data will be avalible for 7 days into the future via API.

### [Featured notebooks](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks)

Read the [project proposal](https://github.com/gperdrizet/wildfire_production/blob/master/project_info/project_proposal.md)
or look at the [project walkthrough](https://github.com/gperdrizet/wildfire_production/blob/master/project_info/project_walkthrough.ipynb).

## Data Sources
1. Historical wildfire activity: United States Department of Agriculture Research Data Archive, [*Spatial wildfire occurrence data for the United States, 1992-2015*](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.4)<sup>1</sup>
2. Historical metrology data: National Oceanic and Atmospheric Administration, [*North American Regional Reanalysis*](https://catalog.data.gov/dataset/ncep-north-american-regional-reanalysis-narr)<sup>2</sup>


### Methods Used

* Machine Learning
* Gradient boosted decision trees
* Deep neural networks
* Parallel long short term memory neural networks
* Cartographic Projection
* Time Series Analysis
* Feature Engineering
* Hyperparameter optimization
* Metaparameter optimization
* Gaussian process optimization
* Cox-Box quantile normalization
* Kolmogorov–Smirnov
* Recursive sample stratification

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

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/))
2. Training data sets are being kept [here](https://www.perdrizet.org/data/wildfire/)
3. Data exploration/transformation notebooks are being kept [here](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks)
4. Finalized scripts for data aquisition and transformation are being kept [here](https://github.com/gperdrizet/wildfire_production/tree/master/training_data_pipeline)
5. Deployment related scripts and data are [here](https://github.com/gperdrizet/wildfire_production/tree/master/deployment)


## Contributing Members

**Team Lead (Contact) : [George Perdrizet](https://github.com/gperdrizet)**

## References
1. Short, Karen C. 2017. *Spatial wildfire occurrence data for the United States, 1992-2015* [FPA_FOD_20170508]. 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.4
2. NCEP Reanalysis data provided by the NOAA/OAR/ESRL PSD, Boulder, Colorado, USA, from their Web site at https://www.esrl.noaa.gov/psd/
