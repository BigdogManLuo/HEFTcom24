import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from utils_forecasting import forecast_wind,forecast_solar
import pickle
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
from tqdm import tqdm
from itertools import chain

np.random.seed(42)

def stackSisterForecast(targetType):

    '''
    targetType: "wind" or "solar"
    '''

    if targetType=="solar":
        #Load Dataset
        Dataset_dwd=pd.read_csv("../data/dataset/train/dwd/SolarDataset.csv")
        Dataset_gfs=pd.read_csv("../data/dataset/train/gfs/SolarDataset.csv")
    elif targetType=="wind":
        #Load Dataset
        Dataset_dwd=pd.read_csv("../data/dataset/train/dwd/WindDataset.csv")
        Dataset_gfs=pd.read_csv("../data/dataset/train/gfs/WindDataset.csv")

    #Extract Features and Labels
    features_dwd=Dataset_dwd.iloc[:,:-1].values
    labels_dwd=Dataset_dwd.iloc[:,-1].values

    features_gfs=Dataset_gfs.iloc[:,:-1].values
    labels_gfs=Dataset_gfs.iloc[:,-1].values

    #Smaller size
    idxs=np.random.choice(features_dwd.shape[0],size=int(0.2*features_dwd.shape[0]),replace=False)
    features_dwd=features_dwd[idxs]
    labels_dwd=labels_dwd[idxs]
    features_gfs=features_gfs[idxs]
    labels_gfs=labels_gfs[idxs]

    #Base Learners
    predictions = {}
    for source in ["dwd","gfs"]:
        if targetType=="wind":
            configs={
                "wind_features":features_dwd if source=="dwd" else features_gfs,
                "full":False,
                "WLimit":False,
                "source":source
            }

            Generation_Forecast=forecast_wind(**configs)
        elif targetType=="solar":
            configs={
                "solar_features":features_dwd if source=="dwd" else features_gfs,
                "full":False,
                "hours":features_dwd[:,-1],
                "SolarRevise":False,
                "rolling_test":False
            }

            Generation_Forecast=forecast_solar(**configs)

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            predictions[f"{source}_{targetType}_q{quantile}"]=Generation_Forecast[f"q{quantile}"]
        
    Predictions = pd.DataFrame(predictions)


    if not os.path.exists("../models/Ensemble/train"):
        os.makedirs("../models/Ensemble/train")
    if not os.path.exists("../models/Ensemble/full"):
        os.makedirs("../models/Ensemble/full")

    Models_meta={}
    for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
        
        features_train=Predictions[[f"dwd_{targetType}_q{quantile}",f"gfs_{targetType}_q{quantile}"]].values
        Models_meta[f"q{quantile}"]=QuantileRegressor(quantile=quantile/100,solver=solver,alpha=0,random_state=42)
        Models_meta[f"q{quantile}"].fit(features_train,labels_dwd)

        with open(f"../models/Ensemble/train/{targetType}_q{quantile}.pkl","wb") as f:
            pickle.dump(Models_meta[f"q{quantile}"],f)
        with open(f"../models/Ensemble/full/{targetType}_q{quantile}.pkl","wb") as f:
            pickle.dump(Models_meta[f"q{quantile}"],f)
        
        print(f"q{quantile}:",Models_meta[f"q{quantile}"].coef_)

if __name__ == "__main__":
    for targetType in ["wind","solar"]:
        stackSisterForecast(targetType)




