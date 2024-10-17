import utils
import pandas as pd
import numpy as np

# Load Data
features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype="latest",source="dwd")
features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype="latest",source="gfs")
IntegratedDataset=pd.read_csv("data/dataset/latest/IntegratedDataset.csv")

# Quantile Add
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "WLimit":True,
    "SolarRevise":True,
    "rolling_test":True,
    "availableCapacities":IntegratedDataset["availableCapacity"].values,
    "aggregation":False
}
Total_generation_forecast0,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)
mpd_total0=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast0)


#Probabilistic Aggregation
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "WLimit":True,
    "SolarRevise":True,
    "rolling_test":True,
    "availableCapacities":IntegratedDataset["availableCapacity"].values,
    "aggregation":True
}
Total_generation_forecast1,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)
mpd_total1=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast1)

print(f"Without Aggregation:{mpd_total0}\nWith Aggregation:{mpd_total1}")

#%% Plotting
utils.plotPowerGeneration(Total_generation_forecast0,labels_solar_dwd+labels_wind_dwd,filename="total_gen_quantile_add.png",x_range0=2400,ptype="total")
utils.plotPowerGeneration(Total_generation_forecast1,labels_solar_dwd+labels_wind_dwd,filename="total_gen_aggregation.png",x_range0=2400,ptype="total")