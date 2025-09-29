import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMRegressor
import pandas as pd
import pickle
from tqdm import tqdm
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
import numpy as np
import utils_forecasting
import utils_data


IntegratedDataset_train=pd.read_csv("../data/dataset/train/IntegratedDataset.csv")


for source in ["dwd","gfs"]:

    columns_wind_features,columns_solar_features=utils_data.getFeaturesName(source)
    
    features=IntegratedDataset_train[columns_wind_features+columns_solar_features]
    labels=IntegratedDataset_train["total_generation_MWh"]
    
    print("Training model for",source)

    for quantile in tqdm(range(10,100,10)):
        params={
            'objective':'quantile',
            'alpha':quantile/100,
            'num_leaves': 500,
            'n_estimators': 2000,
            'max_depth':6,
            'min_data_in_leaf': 200,
            'learning_rate':0.2,
            'lambda_l1': 40,           
            'lambda_l2': 80,
            'random_state':42,
            'verbose':-1
            }
        model=LGBMRegressor(**params)
        model.fit(features,labels)
        if not os.path.exists(f"../models/benchmark/train/{source}"):
            os.makedirs(f"../models/benchmark/train/{source}")
        with open(f"../models/benchmark/train/{source}/quantile_{quantile}.pkl","wb") as f:
            pickle.dump(model,f)

#%% Stacking

#Load Dataset
columns_wind_features_dwd,columns_solar_features_dwd=utils_data.getFeaturesName(source="dwd")
columns_wind_features_gfs,columns_solar_features_gfs=utils_data.getFeaturesName(source="gfs")
features_dwd=IntegratedDataset_train[columns_wind_features_dwd+columns_solar_features_dwd]
features_gfs=IntegratedDataset_train[columns_wind_features_gfs+columns_solar_features_gfs]
labels=IntegratedDataset_train["total_generation_MWh"]

#Smaller size
idxs=np.random.choice(features_dwd.shape[0],size=int(0.1*features_dwd.shape[0]),replace=False)
features_dwd=features_dwd.iloc[idxs]
features_gfs=features_gfs.iloc[idxs]
labels=labels.iloc[idxs]

#Base Learners
predictions = {}
for source in ["dwd","gfs"]:

    features=features_dwd if source=="dwd" else features_gfs

    total_generation_forecast=utils_forecasting.forecastTotalByBenchmark(features,source=source)
    for quantile in range(10,100,10):
        predictions[f"{source}_total_q{quantile}"]=total_generation_forecast[f"q{quantile}"]

Predictions = pd.DataFrame(predictions)

Models_meta={}
for quantile in tqdm(range(10,100,10)):

    features_train=Predictions[[f"dwd_total_q{quantile}",f"gfs_total_q{quantile}"]].values
    Models_meta[f"q{quantile}"]=QuantileRegressor(quantile=quantile/100,solver=solver,alpha=0,random_state=42)
    Models_meta[f"q{quantile}"].fit(features_train,labels)

    if not os.path.exists("../models/benchmark/train/ensemble/"):
        os.makedirs("../models/benchmark/train/ensemble/")
    with open(f"../models/benchmark/train/ensemble/quantile_{quantile}.pkl","wb") as f:
        pickle.dump(Models_meta[f"q{quantile}"],f)

    print(f"q{quantile}:",Models_meta[f"q{quantile}"].coef_)


