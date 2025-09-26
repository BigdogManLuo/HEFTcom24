import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import utils_forecasting
import utils

# Load Data
IntegratedDataset=pd.read_csv("../data/dataset/latest/IntegratedDataset.csv")

def testEnsemble(ftype,pathtype):
    features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype=pathtype,source="dwd")
    features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype=pathtype,source="gfs")

    if ftype=="wind":
        if pathtype=="test":
            params_dwd={
            "wind_features":features_wind_dwd,
            "full":False,
            "WLimit":False,
            "source":"dwd"
            }
            labels=labels_wind_dwd

            params_gfs={
                "wind_features":features_wind_gfs,
                "full":False,
                "WLimit":False,
                "source":"gfs"
            }

            params_stacking={
                "wind_features_dwd":features_wind_dwd,
                "wind_features_gfs":features_wind_gfs,
                "full":False,
                "WLimit":False,
            }
            
        if pathtype=="latest":

            params_dwd={
                "wind_features":features_wind_dwd,
                "full":True,
                "WLimit":True,
                "availableCapacities":IntegratedDataset["availableCapacity"].values,
                "source":"dwd"
            }
            
            labels=labels_wind_dwd

            params_gfs={
                "wind_features":features_wind_gfs,
                "full":True,
                "WLimit":True,
                "availableCapacities":IntegratedDataset["availableCapacity"].values,
                "source":"gfs"
            }

            params_stacking={
                "wind_features_dwd":features_wind_dwd,
                "wind_features_gfs":features_wind_gfs,
                "full":True,
                "WLimit":True,
                "availableCapacities":IntegratedDataset["availableCapacity"].values
            }


        Generation_Forecast_dwd=utils_forecasting.forecast_wind(**params_dwd)
        Generation_Forecast_gfs=utils_forecasting.forecast_wind(**params_gfs)
        Generation_Forecast_stacking=utils_forecasting.forecast_wind_ensemble(**params_stacking)

    if ftype=="solar":
        if pathtype=="test":
            params_dwd={
                "solar_features":features_solar_dwd,
                "hours":features_solar_dwd[:,-1],
                "full":False,
                "SolarRevise":False,
                "rolling_test":False,
                "source":"dwd"
            }
            labels=labels_solar_dwd

            params_gfs={
                "solar_features":features_solar_gfs,
                "hours":features_solar_gfs[:,-1],
                "full":False,
                "SolarRevise":False,
                "rolling_test":False,
                "source":"gfs"
            }

            params_stacking={
                "solar_features_dwd":features_solar_dwd,
                "solar_features_gfs":features_solar_gfs,
                "hours":features_solar_dwd[:,-1],
                "full":False,
                "SolarRevise":False,
            }


        if pathtype=="latest":
            params_dwd={
            "solar_features":features_solar_dwd,
            "hours":features_solar_dwd[:,-1],
            "full":True,
            "SolarRevise":True,
            "rolling_test":True,
            "source":"dwd"
            }

            labels=labels_solar_dwd

            params_gfs={
                "solar_features":features_solar_gfs,
                "hours":features_solar_gfs[:,-1],
                "full":True,
                "SolarRevise":True,
                "rolling_test":True,
                "source":"gfs"
            }

            params_stacking={
                "solar_features_dwd":features_solar_dwd,
                "solar_features_gfs":features_solar_gfs,
                "hours":features_solar_dwd[:,-1],
                "full":True,
                "SolarRevise":True,
            }

        Generation_Forecast_dwd=utils_forecasting.forecast_solar(**params_dwd)
        Generation_Forecast_gfs=utils_forecasting.forecast_solar(**params_gfs)
        Generation_Forecast_stacking=utils_forecasting.forecast_solar_ensemble(**params_stacking)

    mpd_dwd=utils.meanPinballLoss(labels,Generation_Forecast_dwd)
    mpd_gfs=utils.meanPinballLoss(labels,Generation_Forecast_gfs)
    mpd_stacking=utils.meanPinballLoss(labels,Generation_Forecast_stacking)

    mCRPS_dwd=utils.getMCRPS(y_true=labels,y_pred=Generation_Forecast_dwd)
    mCRPS_gfs=utils.getMCRPS(y_true=labels,y_pred=Generation_Forecast_gfs)
    mCRPS_stacking=utils.getMCRPS(y_true=labels,y_pred=Generation_Forecast_stacking)

    mWS_dwd=utils.getWinklerScore(y_true=labels,y_pred=Generation_Forecast_dwd)
    mWS_gfs=utils.getWinklerScore(y_true=labels,y_pred=Generation_Forecast_gfs)
    mWS_stacking=utils.getWinklerScore(y_true=labels,y_pred=Generation_Forecast_stacking)

    CP_dwd=utils.getCoverageProbability(y_true=labels,y_pred=Generation_Forecast_dwd)
    CP_gfs=utils.getCoverageProbability(y_true=labels,y_pred=Generation_Forecast_gfs)
    CP_stacking=utils.getCoverageProbability(y_true=labels,y_pred=Generation_Forecast_stacking)

    print("==================================")
    print("Type:",ftype,"Case","CaseI" if pathtype=="test" else "CaseII")
    print("==================================")
    print(f" DWD MPD:{mpd_dwd}\ GFS MPD:{mpd_gfs}\ Ensemble MPD:{mpd_stacking}")
    print(f" DWD mCRPS:{mCRPS_dwd}\ GFS mCRPS:{mCRPS_gfs}\ Ensemble mCRPS:{mCRPS_stacking}")
    print(f" DWD mWS:{mWS_dwd}\ GFS mWS:{mWS_gfs}\ Ensemble mWS:{mWS_stacking}")
    print(f" DWD CP:{CP_dwd}\ GFS CP:{CP_gfs}\ Ensemble CP:{CP_stacking}")

if __name__ == "__main__":
    for ftype in ["wind","solar"]:
        for pathtype in ["test","latest"]: 
            testEnsemble(ftype,pathtype)