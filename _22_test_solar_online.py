import utils


# Load Data
_,_,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype="latest",source="dwd")
_,_,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype="latest",source="gfs")

#Without Online Post-Processing
params={
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "SolarRevise":False,
    "rolling_test":False
}

Solar_Generation_Forecast0=utils.forecast_solar(**params)
mpd_solar0=utils.meanPinballLoss(labels_solar_dwd,Solar_Generation_Forecast0)

#With Online Post-Processing
params={
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "SolarRevise":True,
    "rolling_test":True
}

Solar_Generation_Forecast1=utils.forecast_solar(**params)
mpd_solar1=utils.meanPinballLoss(labels_solar_dwd,Solar_Generation_Forecast1)

print(f"Without Online Post-Processing:{mpd_solar0}\nWith Online Post-Processing:{mpd_solar1}")


utils.plotPowerGeneration(Solar_Generation_Forecast0,labels_solar_dwd,filename="forecast_origin.png",x_range0=720)
utils.plotPowerGeneration(Solar_Generation_Forecast1,labels_solar_dwd,filename="forecast_revised.png",x_range0=720)