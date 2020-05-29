## California Wildfire Risk Prediction

### Purpose and scope

Wildfires are common, deadly and destructive natural disasters in California. The total economic cost of wildfire damage in 2018 is estimated at 400 billion dollars (1). There are two major wildfire risk assessment systems in the US: the National Weather Service (NWS) red flag warning system (2) and The National Fire Danger Rating System (NFDRS)(3). Both can be improved upon.

The red flag warning system issues a warning when three risk indicators reach a specific threshold: less than 8% fuel moisture, less than 25% relative humidity and winds greater than 15 mph (2). No data for correlation between red flag warnings and actual fires could be found.
The NFDRS is a geospatially resolved fire risk prediction model which assesses wildfire risk in one of five categories. The model comprises a set of knowledge based functions which use several different wildfire fuel models and weather data to classify fire risk (3). NFDRS is more technologically modern than the red flag system, but user intervention and interpretation is still necessary. Many model parameters are subjective and location specific. Again, no information evaluating the model against actual wildfire data could be found.
Neither of the systems incorporate historical wildfire data. Wildfire risk prediction is an excellent candidate for the successful application of modern, supervised machine learning.

For my capstone project I will design and implement a wildfire risk prediction system. The system will use a neural net to predict geospatially resolved wildfire risk in California. This is an excellent capstone project because wildfire prediction is a real world problem of significant social and economic impact. Also, there is room for considerable improvement over current systems via the application of data engineering and machine learning. Furthermore, designing and implementing the data pipeline, predictive model and live risk assessment service will give me the skills I need to start a career as a machine learning engineer.

### Strategy

This project will utilize two types of data: actual historical wildfire locations and weather data. Real time and historical weather data will be acquired from the California Data Exchange Center (CDEC) via web API (4). Live data pertaining to currently active fires will be obtained from the United States Geological Survey (USGS) also via web API (5). A static database of historical wildfire data is available in the United States Department of Agriculture (USDA) research data archive as a downloadable SQLite database (6).
The model will be trained on historical fire and weather data and then used to predict the current risk of fire from real time weather data. Because the locations of all major fires between 1992 and 2015 are recorded in the USDA historical fire data set, this is a supervised machine learning problem. In general the strategy will be:

1. Create training data set – California will be binned by latitude and longitude. The time period spanned by the USDA historical wildfire data set will be broken up into time bins. Each time and location will then be assigned values for the weather variables of interest and the presence or absence of fire.
2. Train model – Several approaches, appropriate to regression, from traditional and deep learning will be evaluated. The specific goal will be to predict fire ignition probability. For example, in the set of locations for which the model predicts a 90% ignition probability, 90% should actually have an ignition event. Input data for prediction will be fire data and weather data. Fire data includes the total number of fires at each location during the reporting period and the time since the last fire. Weather data includes several variables including: relative humidity, temperature, wind speed, precipitation etc.
3. Deploy model – The user facing final product will take the form of a live fire risk map website and an API to access the data. This data product will retrieve real time weather data from the CDEC to predict current wildfire risk in California.

### Resources

Size estimates for the data sets are as follows:
1. Weather data – data from ~2500 monitoring stations is available through CDEC, ~800 of these stations report data relevant to this project. Retrieving one month of observations from all sensors of interest results in just under 2 million unique observations and occupies 120M on disk. Based on this, weather data for the 23 years in which wildfire data is available will occupy about 5.8G on disk.
2. Historical fire data - just under 2 million unique fires from 1992 to 2015, occupies 790M on disk.
Total available local resources: 44 threads (22 physical cores) 80G memory. The project will be developed with docker to make it easily deployable to a cloud service for training or production if more resources are needed.

### References

1. Myers, J. N. CEO. (2019). [AccuWeather predicts 2018 wildfires will cost California total economic losses of $400 billion](https://www.accuweather.com/en/weather-news/accuweather-predicts-2018-wildfires-will-cost-california-total-economic-losses-of-400-billion/70006691). AccuWeather. Retrieved 2019-10-02.
2. [What is a Red Flag Warning?](https://www.weather.gov/media/lmk/pdf/what_is_a_red_flag_warning.pdf) (PDF). National Weather Service. Retrieved 2019-10-02.
3. [The NFDRS 2016 Models](https://sites.google.com/firenet.gov/nfdrs/the-models). National WildFire Coordinating Group. Retrieved 2019-10-02.
4. [API Documents Web-Services Data Download APPs](https://cdec.water.ca.gov/reportapp/javareports?name=WebWSAppsAPIDocs.pdf) (PDF). California Data Exchange Center. Retrieved 2019-09-22.
5. [Wildfire: GEOMAC](https://www.geomac.gov/). United States Geological Survey. Retrieved 2019-09-20.
6. Short, K. C. (2017) [Spatial wildfire occurrence data for the United States, 1992-2015](https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.4). 4th Edition. Fort Collins, CO: Forest Service Research Data Archive. Retrieved 2019-09-22.

