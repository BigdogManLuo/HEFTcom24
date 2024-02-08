import pandas as pd
import numpy as np
import pickle


#%% 合并两个天气源的数据集

dwd_dataset=pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")
gfs_dataset=pd.read_csv("data/dataset/gfs/IntegratedDataset.csv")

#将dwd_dataset和gfs_dataset按照ref_datetime和valid_datetime合并
IntegratedDataset=pd.merge(dwd_dataset,gfs_dataset,on=["ref_datetime","valid_datetime"])


#%% 分别构造风电、光伏数据集
columns_wind_features=["ws_100_t-1_dwd_1","ws_100_t_dwd_1","ws_100_t+1_dwd_1","ws_100_t-1_gfs_1","ws_100_t_gfs_1","ws_100_t+1_gfs_1"]
columns_wind_labels=["Wind_MWh_credit_x"]
columns_solar_features=["rad_t-1_dwd","rad_t_dwd","rad_t+1_dwd","rad_t-1_gfs","rad_t_gfs","rad_t+1_gfs"]
columns_solar_labels=["Solar_MWh_credit_x"]

WindDataset=IntegratedDataset[columns_wind_features+columns_wind_labels]
SolarDataset=IntegratedDataset[columns_solar_features+columns_solar_labels]

#记录均值方差特征
Mean_features_wind=WindDataset.iloc[:,:-1].mean()
Mean_labels_wind=WindDataset.iloc[:,-1].mean()
Std_features_wind=WindDataset.iloc[:,:-1].std()
Std_labels_wind=WindDataset.iloc[:,-1].std()

Mean_features_solar=SolarDataset.iloc[:,:-1].mean()
Mean_labels_solar=SolarDataset.iloc[:,-1].mean()
Std_features_solar=SolarDataset.iloc[:,:-1].std()
Std_labels_solar=SolarDataset.iloc[:,-1].std()

Dataset_stats = {
    "Mean": {
        "features": {
            "wind": Mean_features_wind,
            "solar": Mean_features_solar
        },
        "labels": {
            "wind": Mean_labels_wind,
            "solar":Mean_labels_solar
        }
    },
    "Std": {
        "features": {
            "wind": Std_features_wind,
            "solar": Std_features_solar
        },
        "labels": {
            "wind": Std_labels_wind,
            "solar":Std_labels_solar
        }
    }
}

#保存Dataset_stats字典到文件
with open('data/dataset/inte/Dataset_stats.pkl', 'wb') as handle:
    pickle.dump(Dataset_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

#加入不需要标准化的特征
hours = IntegratedDataset["hours_x"].copy()
SolarDataset.insert(len(SolarDataset.columns) - 1, "hours", hours)

hours_diff=IntegratedDataset["hours_diff_x"].copy()
SolarDataset.insert(len(SolarDataset.columns) - 1, "hours_diff", hours_diff)
WindDataset.insert(len(WindDataset.columns) - 1, "hours_diff", hours_diff)

#保存数据集
IntegratedDataset.to_csv("data/dataset/inte/IntegratedDataset.csv",index=False)
WindDataset.to_csv("data/dataset/inte/WindDataset.csv",index=False)
SolarDataset.to_csv("data/dataset/inte/SolarDataset.csv",index=False)
