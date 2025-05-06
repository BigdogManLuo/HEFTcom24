import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMRegressor
import pandas as pd
import pickle
from tqdm import tqdm

#加载数据集
IntegratedDataset_train=pd.read_csv("../data/dataset/train/IntegratedDataset.csv")

#移除ref_datetime,valid_datetime
IntegratedDataset_train.drop(columns=["ref_datetime","valid_datetime","Wind_MWh_credit","Solar_MWh_credit","Wind_MWh_credit_y","Solar_MWh_credit_y","total_generation_MWh_y","DA_Price_y","SS_Price_y","DA_Price","SS_Price","hours_y"],inplace=True)

#提取特征和标签
features=IntegratedDataset_train.drop(columns=["total_generation_MWh"])
labels=IntegratedDataset_train["total_generation_MWh"]
features_name=features.columns

#训练模型
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
        'verbose':-1
        }
    model=LGBMRegressor(**params)
    model.fit(features,labels)
    with open(f"../models/benchmark/train/quantile_{quantile}.pkl","wb") as f:
        pickle.dump(model,f)
        
