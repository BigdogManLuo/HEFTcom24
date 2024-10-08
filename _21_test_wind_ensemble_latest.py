import pandas as pd
import utils


# Load Data
features_wind_dwd,labels_wind_dwd,_,_=utils.loadFeaturesandLabels(pathtype="latest",source="dwd")
features_wind_gfs,labels_wind_gfs,_,_=utils.loadFeaturesandLabels(pathtype="latest",source="gfs")
IntegratedDataset=pd.read_csv("data/dataset/latest/IntegratedDataset.csv")

#dwd only
params={
    "wind_features":features_wind_dwd,
    "full":True,
    "WLimit":True,
    "availableCapacities":IntegratedDataset["availableCapacity"].values,
    "source":"dwd"
}
Wind_Generation_Forecast=utils.forecast_wind(**params)

mpd_wind_dwd=utils.meanPinballLoss(labels_wind_dwd,Wind_Generation_Forecast)

#gfs only
params={
    "wind_features":features_wind_gfs,
    "full":True,
    "WLimit":True,
    "availableCapacities":IntegratedDataset["availableCapacity"].values,
    "source":"gfs"
}
Wind_Generation_Forecast=utils.forecast_wind(**params)
mpd_wind_gfs=utils.meanPinballLoss(labels_wind_gfs,Wind_Generation_Forecast)

#dwd+gfs
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "full":True,
    "WLimit":True,
    "availableCapacities":IntegratedDataset["availableCapacity"].values
}
Wind_Generation_Forecast=utils.forecast_wind_ensemble(**params)
mpd_wind_ensemble=utils.meanPinballLoss(labels_wind_dwd,Wind_Generation_Forecast)

print(f"Wind DWD MPD:{mpd_wind_dwd}\nWind GFS MPD:{mpd_wind_gfs}\nWind Ensemble MPD:{mpd_wind_ensemble}")









