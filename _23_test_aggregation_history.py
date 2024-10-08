import utils

# Load Data
features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype="test",source="dwd")
features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype="test",source="gfs")

# Quantile Add
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":False,
    "WLimit":False,
    "SolarRevise":False,
    "aggregation":False
}
Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)
mpd_total0=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast)

#Probabilistic Aggregation
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":False,
    "WLimit":False,
    "SolarRevise":False,
    "aggregation":True
}
Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)
mpd_total1=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast)

print(f"Without Aggregation:{mpd_total0}\nWith Aggregation:{mpd_total1}")
