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

def stackSisterForecast(is_full):
    if is_full:
        path="full"
    else:
        path="train"

    #Load Dataset
    WindDataset_dwd=pd.read_csv(f"data/dataset/{path}/dwd/WindDataset.csv")
    WindDataset_gfs=pd.read_csv(f"data/dataset/{path}/gfs/WindDataset.csv")

    #Extract Features and Labels
    features_wind_dwd=WindDataset_dwd.iloc[:,:-1].values
    labels_wind_dwd=WindDataset_dwd.iloc[:,-1].values

    features_wind_gfs=WindDataset_gfs.iloc[:,:-1].values
    labels_wind_gfs=WindDataset_gfs.iloc[:,-1].values

    #Base Learners
    predictions = {}
    for source in ["dwd","gfs"]:

        configs={
            "wind_features":features_wind_dwd if source=="dwd" else features_wind_gfs,
            "full":is_full,
            "WLimit":False,
            "source":source
        }

        Wind_Generation_Forecast=forecast_wind(**configs)

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            predictions[f"{source}_wind_q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]
        
    Predictions = pd.DataFrame(predictions)

    #Meta Learner
    idxs=np.random.choice(Predictions.index,size=int(0.1*len(Predictions)),replace=False)
    Predictions_train=Predictions.loc[idxs]
    labels_train=labels_wind_gfs[idxs]

    if not os.path.exists(f"models/Ensemble/{path}"):
        os.makedirs(f"models/Ensemble/{path}")

    Models_meta={}
    for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
        
        features_train=Predictions_train[[f"dwd_wind_q{quantile}",f"gfs_wind_q{quantile}"]].values
        Models_meta[f"q{quantile}"]=QuantileRegressor(quantile=quantile/100,solver=solver,alpha=0)
        Models_meta[f"q{quantile}"].fit(features_train,labels_train)

        with open(f"models/Ensemble/{path}/wind_q{quantile}.pkl","wb") as f:
            pickle.dump(Models_meta[f"q{quantile}"],f)
        
        print(f"q{quantile}:",Models_meta[f"q{quantile}"].coef_)

if __name__ == "__main__":
    stackSisterForecast(is_full=False)
    stackSisterForecast(is_full=True)





