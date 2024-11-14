import numpy as np
import pandas as pd
from utils import forecast_solar
import pickle
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
from tqdm import tqdm

#Load Training Data
SolarDataset_dwd=pd.read_csv("data/dataset/train/dwd/SolarDataset.csv")
SolarDataset_gfs=pd.read_csv("data/dataset/train/gfs/SolarDataset.csv")

#Extract Features and Labels
features_solar_dwd=SolarDataset_dwd.iloc[:,:-1].values
labels_solar_dwd=SolarDataset_dwd.iloc[:,-1].values

features_solar_gfs=SolarDataset_gfs.iloc[:,:-1].values
labels_solar_gfs=SolarDataset_gfs.iloc[:,-1].values

#Predict by Base Learners
Predictions=pd.DataFrame()
for source in ["dwd","gfs"]:

    configs={
        "solar_features":features_solar_dwd if source=="dwd" else features_solar_gfs,
        "hours":SolarDataset_dwd["hours"].values if source=="dwd" else SolarDataset_gfs["hours"].values,
        "full":False,
        "model_name":"LGBM",
        "source":source,
        "SolarRevise":False,
    }

    Solar_Generation_Forecast=forecast_solar(**configs)

    for quantile in range(10,100,10):
        Predictions[f"{source}_solar_q{quantile}"]=Solar_Generation_Forecast[f"q{quantile}"]


#Train Meta Learner
idxs=np.random.choice(Predictions.index,size=int(0.1*len(Predictions)),replace=False)
Predictions_train=Predictions.loc[idxs]
labels_train=labels_solar_gfs[idxs]

Models_meta={}
for quantile in tqdm(range(10,100,10)):
    features_train=Predictions_train[[f"dwd_solar_q{quantile}",f"gfs_solar_q{quantile}"]].values
    Models_meta[f"q{quantile}"]=QuantileRegressor(quantile=quantile/100,solver=solver,alpha=0)
    Models_meta[f"q{quantile}"].fit(features_train,labels_train)

    #Save Model
    with open(f"models/Ensemble/train/solar_q{quantile}.pkl","wb") as f:
        pickle.dump(Models_meta[f"q{quantile}"],f)
    
    print(f"q{quantile}:",Models_meta[f"q{quantile}"].coef_)
