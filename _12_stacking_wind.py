import numpy as np
import pandas as pd
from utils import forecast_wind
import pickle
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
from tqdm import tqdm
from itertools import chain
import os 

np.random.seed(42)

def stackSisterForecast():
    
    #Load Dataset
    WindDataset_dwd=pd.read_csv("data/dataset/train/dwd/WindDataset.csv")
    WindDataset_gfs=pd.read_csv("data/dataset/train/gfs/WindDataset.csv")

    #Extract Features and Labels
    features_wind_dwd=WindDataset_dwd.iloc[:,:-1].values
    labels_wind_dwd=WindDataset_dwd.iloc[:,-1].values

    features_wind_gfs=WindDataset_gfs.iloc[:,:-1].values
    labels_wind_gfs=WindDataset_gfs.iloc[:,-1].values

    #Smaller size
    idxs=np.random.choice(features_wind_dwd.shape[0],size=int(0.2*features_wind_dwd.shape[0]),replace=False)
    features_wind_dwd=features_wind_dwd[idxs]
    labels_wind_dwd=labels_wind_dwd[idxs]
    features_wind_gfs=features_wind_gfs[idxs]
    labels_wind_gfs=labels_wind_gfs[idxs]

    #Base Learners
    predictions = {}
    for source in ["dwd","gfs"]:

        configs={
            "wind_features":features_wind_dwd if source=="dwd" else features_wind_gfs,
            "full":False,
            "WLimit":False,
            "source":source
        }

        Wind_Generation_Forecast=forecast_wind(**configs)

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            predictions[f"{source}_wind_q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]
        
    Predictions = pd.DataFrame(predictions)


    if not os.path.exists("models/Ensemble/train"):
        os.makedirs("models/Ensemble/trian")
    if not os.path.exists("models/Ensemble/full"):
        os.makedirs("models/Ensemble/full")

    Models_meta={}
    for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
        
        features_train=Predictions[[f"dwd_wind_q{quantile}",f"gfs_wind_q{quantile}"]].values
        Models_meta[f"q{quantile}"]=QuantileRegressor(quantile=quantile/100,solver=solver,alpha=0)
        Models_meta[f"q{quantile}"].fit(features_train,labels_wind_dwd)

        with open(f"models/Ensemble/train/wind_q{quantile}.pkl","wb") as f:
            pickle.dump(Models_meta[f"q{quantile}"],f)
        with open(f"models/Ensemble/full/wind_q{quantile}.pkl","wb") as f:
            pickle.dump(Models_meta[f"q{quantile}"],f)
        
        print(f"q{quantile}:",Models_meta[f"q{quantile}"].coef_)

if __name__ == "__main__":
    stackSisterForecast()




