import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils_forecasting
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

Solar_Generation_Forecast0=utils_forecasting.forecast_solar(**params)
mpd_solar0=utils.meanPinballLoss(labels_solar_dwd,Solar_Generation_Forecast0)
pl_solar0=utils.getPinballLosses(labels_solar_dwd,Solar_Generation_Forecast0)
mCRPS_solar0=utils.getMCRPS(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast0)
mWS_solar0=utils.getWinklerScore(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast0)
mCP0=utils.getCoverageProbability(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast0)

# Offline Post-Processing
params={
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "SolarRevise":True,
    "rolling_test":False
}

Solar_Generation_Forecast_offlinePP=utils_forecasting.forecast_solar(**params)
mpd_solar_offlinePP=utils.meanPinballLoss(labels_solar_dwd,Solar_Generation_Forecast_offlinePP)
pl_solar_offlinePP=utils.getPinballLosses(labels_solar_dwd,Solar_Generation_Forecast_offlinePP)
mCRPS_solar_offlinePP=utils.getMCRPS(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_offlinePP)
mWS_solar_offlinePP=utils.getWinklerScore(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_offlinePP)
mCP_offlinePP=utils.getCoverageProbability(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_offlinePP)

#Online Post-Processing
params={
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":True,
    "SolarRevise":True,
    "rolling_test":True
}

Solar_Generation_Forecast_onlinePP=utils_forecasting.forecast_solar(**params)
mpd_solar_onlinePP=utils.meanPinballLoss(labels_solar_dwd,Solar_Generation_Forecast_onlinePP)
pl_solar_onlinePP=utils.getPinballLosses(labels_solar_dwd,Solar_Generation_Forecast_onlinePP)
mCRPS_solar_onlinePP=utils.getMCRPS(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_onlinePP)
mWS_solar_onlinePP=utils.getWinklerScore(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_onlinePP)
mCP_onlinePP=utils.getCoverageProbability(y_true=labels_solar_dwd,y_pred=Solar_Generation_Forecast_onlinePP)

print("===========================Pinball Loss===========================")
print(f"Without Online Post-Processing:{mpd_solar0}\n Offline Post-Processing:{mpd_solar_offlinePP}\n Online Post-Processing:{mpd_solar_onlinePP}")
print("===========================Mean Continuous Ranked Probability Score===========================")
print(f"Without Online Post-Processing:{mCRPS_solar0}\n Offline Post-Processing:{mCRPS_solar_offlinePP}\n Online Post-Processing:{mCRPS_solar_onlinePP}")
print("===========================Winkler Score===========================")
print(f"Without Online Post-Processing:{mWS_solar0}\n Offline Post-Processing:{mWS_solar_offlinePP}\n Online Post-Processing:{mWS_solar_onlinePP}")
print("===========================Coverage Probability===========================")
print(f"Without Online Post-Processing:{mCP0}\n Offline Post-Processing:{mCP_offlinePP}\n Online Post-Processing:{mCP_onlinePP}")

utils.plotPowerGeneration(Solar_Generation_Forecast0,labels_solar_dwd,filename="forecast_origin.png",x_range0=720)
utils.plotPowerGeneration(Solar_Generation_Forecast_onlinePP,labels_solar_dwd,filename="forecast_revised.png",x_range0=720)
