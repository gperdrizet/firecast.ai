# Firecast.ai - machine learning wildfire risk forecasting

![Heatmap banner](https://github.com/gperdrizet/firecast.ai/blob/master/project_info/figures/heatmap_cropped.png?raw=true)

The goal of this project is to build a machine learning model which can predict wildfire ignition risk in California from publicly available meteorology and fire activity data.

#### -- Project Status: [Active]

## Project Description

Wild fires are common, destructive and deadly natural disasters. Current meteorology based risk prediction methods can be improved upon by: 

1. The application of modern machine learning techniques
2. Use of historical wildfire data for model training and validation

This project uses a parallel LSTM neural network to predict geospatialy resolved wildfire ignition risk in California. The model was trained on a combined dataset produced from the USDA historical wildfire activity dataset(1) and meterological data from NOAA's North American Regional Reanalysis(2). This project is currently in the deployment phase. Live prediction data will be avalible for 7 days into the future via API. For more background information please see the full [project proposal](https://github.com/gperdrizet/wildfire_production/tree/master/project_info/project_proposal.md)

## Using this repository
First, clone the repo:

    git clone https://github.com/gperdrizet/firecast.ai.git
    
Next, you have two options to install required packages:

1. Using conda (reccomended). This will install a complete copy of the development environment, including all dependencies.
    
    cd firecast.ai
    conda env create -f environment.yml
    
2. Using pip and venv.

    python3 -m venv firecast.ai
    source firecast.ai/bin/activate
    cd firecast.ai
    pip install -r requirements.txt

## Featured notebooks
1. [Exploratory data analysis](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/01-exploratory_data_analysis.ipynb)
2. [Classifier model evaluation](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/02-classifier_model_selection.ipynb)
3. [Feature engineering](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/03-add_features.ipynb)
4. [Fully stratified sampling](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/04-recursive_sampling.ipynb)
5. [XGBoost optimization](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/05-XGBoost_optimization.ipynb)
6. [Deep neural network optimization](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/06-deep_neural_network_optimization.ipynb)
7. [Single LSTM optimization](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/07-single_LSTM_optimization.ipynb)
8. [Geospatialy parallel LSTM](https://github.com/gperdrizet/wildfire_production/tree/master/notebooks/08-parallel_LSTM.ipynb)

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
