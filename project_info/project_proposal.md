## California Wildfire Risk Prediction

### Purpose and scope

Wildfires are common, deadly, and destructive natural disasters in California. The total economic cost of wildfire damage in 2018 is estimated at 400 billion dollars(1). There are two major wildfire risk assessment systems in the US: the National Weather Service (NWS) red flag warning system(2) and The National Fire Danger Rating System (NFDRS)(3). Historical wildfire data is not considered in either of these systems. The goal of this project is to build a system which predicts geospatially resolved wildfire ignition risk in California using a machine learning algorithm trained on historical wildfire ignition and weather data. Once trained, the algorithm will use live weather predictions to forecast wildfire ignition risk. Predictions will be made publicly available via an API.

### Background

Wildfire risk prediction is an excellent candidate for the successful application of modern, supervised machine learning.

The NWS issues a red flag warning when three risk indicators reach a specific threshold: less than 8% fuel moisture, less than 25% relative humidity and winds greater than 15 mph(2). No historical data addressing correlation between red flag warnings and wildfires could be found.

The NFDRS is a geospatially resolved fire risk prediction model which assigns wildfire risk to one of five categories. The model comprises a set of knowledge-based functions which use several wildfire fuel models and weather data to classify wildfire risk(3). The NFDRS is more technologically modern than the NWS red flag system, but user intervention and interpretation is still necessary. Many model parameters are subjective and location specific. No information evaluating the model against actual wildfire data could be found.

### Strategy
This project will utilize three types of data: past and future weather data and historical wildfire data. Historical weather data will be acquired from the from the National Oceanic and Atmospheric Administration (NOAA) via web API(4). Live weather prediction data is available via OpenWeather’s One Call API(5). A static database of historical wildfire data is available in the United States Department of Agriculture (USDA) research data archive as a downloadable SQLite database(6).
The model will be trained on historical wildfire and weather data and then used to predict wildfire ignition risk from real time weather data. Because the locations of all major wildfires between 1992 and 2015 are recorded in the USDA historical wildfire data set, this is a supervised machine learning problem. In general the strategy will be:

1. Create training data set – California will be binned by latitude and longitude. The time period spanned by the USDA historical wildfire data set will be broken up into time bins. Each time and location will then be assigned values for the weather variables of interest and the presence or absence of fire.
2. Train model – Several approaches, appropriate to regression, from traditional and deep learning will be evaluated. The specific goal will be to predict fire ignition probability. For example, in the set of locations for which the model predicts a 90% ignition probability, 90% should actually have an ignition event. Input data for prediction will be fire data and weather data. Fire data includes the total number of fires at each location during the reporting period and the time since the last fire. Weather data includes several variables including: relative humidity, temperature, wind speed, precipitation etc.
3. Deploy model – Once trained, the model will be deployed as part of a pipeline which ingests weather forecast data and predicts geospatially resolved wildfire risk. Live predictions will be updated daily and made available via an API.

### Resources

Size estimates for the data sets are as follows:
1. Historical weather data – data for 410 geospatial bins In California. The initial pass will include any weather variable which could possibly be relevant spanning 1992 to 2015 at a three hour resolution. Raw data occupies 91GB on disk.
2. Historical fire data - just under 2 million unique fires from 1992 to 2015. Raw data occupies 760MB on disk.

Total available local resources: Beowulf style research compute cluster with 7 nodes, 40 threads (20 physical cores) 96G memory, Nvidia GTX1070, 8GB, 1TB solid state dedicated data storage.

### References

1. Myers, J. N. CEO. (2019). [AccuWeather predicts 2018 wildfires will cost California total economic losses of $400 billion](https://www.accuweather.com/en/weather-news/accuweather-predicts-2018-wildfires-will-cost-california-total-economic-losses-of-400-billion/70006691). AccuWeather. Retrieved 2019-10-02.
2. [What is a Red Flag Warning?](https://www.weather.gov/media/lmk/pdf/what_is_a_red_flag_warning.pdf) (PDF). National Weather Service. Retrieved 2019-10-02.
3. [The NFDRS 2016 Models](https://sites.google.com/firenet.gov/nfdrs/the-models). National WildFire Coordinating Group. Retrieved 2019-10-02.
4. Mesinger, F., G. DiMego, E. Kalnay, K. Mitchell, et al, (2006): [North American Regional Reanalysis](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-regional-reanalysis-narr). Bulletin of the American Meteorological Society, 87, 343–360, doi:10.1175/BAMS-87-3-343.
5. [OpenWeather One Call API.](https://openweathermap.org/api/one-call-api) OpenWeatherMap.og. Retrieved 2019-09-20.
6. Short, K. C. (2017) [Spatial wildfire occurrence data for the United States, 1992-2015](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.4). 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. Retrieved 2019-09-22.

