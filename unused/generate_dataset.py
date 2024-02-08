import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import comp_utils

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pickle as pkl
from tqdm import tqdm

#%% 风电数据处理
dwd_Hornsea1_old = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_new = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
gfc_Hornsea1=xr.open_dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")

#选取特征
dwd_Hornsea1_features=dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features=dwd_Hornsea1_features.merge(dwd_Hornsea1["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
dwd_Hornsea1_features=dwd_Hornsea1_features.merge(dwd_Hornsea1["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),
    how="outer",on=["ref_datetime","valid_datetime"])

gfc_Hornsea1_features=gfc_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
gfc_Hornsea1_features=gfc_Hornsea1_features.merge(gfc_Hornsea1["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
gfc_Hornsea1_features=gfc_Hornsea1_features.merge(gfc_Hornsea1["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),
    how="outer",on=["ref_datetime","valid_datetime"])


#标准化时间
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

gfc_Hornsea1_features["ref_datetime"] = gfc_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
gfc_Hornsea1_features["valid_datetime"] = gfc_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(gfc_Hornsea1_features["valid_datetime"],unit="hours")

#使用最新的天气预报修正作为特征

#1.提取ref_datetime为2023-10-27 00:00:00的数据
dwd_Hornsea1_feature_tmp = dwd_Hornsea1_features[dwd_Hornsea1_features["ref_datetime"] == "2023-10-27 00:00:00"]

#2. 提取valid_datetime为2023-10-27 06:00:00——2023-10-28 00:00:00 的数据
dwd_Hornsea1_feature_tmp = dwd_Hornsea1_feature_tmp[dwd_Hornsea1_feature_tmp["valid_datetime"] >= "2023-10-27 06:00:00"]
dwd_Hornsea1_feature_tmp = dwd_Hornsea1_feature_tmp[dwd_Hornsea1_feature_tmp["valid_datetime"] <= "2023-10-28 00:00:00"]

#3. 删除非最新的天气预报数据
dwd_Hornsea1_features = dwd_Hornsea1_features[dwd_Hornsea1_features["valid_datetime"] - dwd_Hornsea1_features["ref_datetime"] < np.timedelta64(6,"h")]

#4. 将最新的天气预报数据与原始数据合并
dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features,dwd_Hornsea1_feature_tmp],axis=0)


#按照valid_datetime，插值为半小时分辨率的数据
dwd_Hornsea1_features = dwd_Hornsea1_features.set_index("valid_datetime").resample("30T").interpolate("linear")
dwd_Hornsea1_features=dwd_Hornsea1_features.reset_index()

#删除最后一条数据
dwd_Hornsea1_features = dwd_Hornsea1_features.iloc[:-1,:]

#清理临时数据
del dwd_Hornsea1_feature_tmp


#%% 光伏数据处理
dwd_solar=xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")

#选取特征
dwd_solar_features=dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features=dwd_solar_features.merge(dwd_solar["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

#标准化时间
dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

#使用最新的天气预报修正作为特征

#1.提取ref_datetime为2023-10-27 00:00:00的数据
dwd_solar_feature_tmp = dwd_solar_features[dwd_solar_features["ref_datetime"] == "2023-10-27 00:00:00"]

#2. 提取valid_datetime为2023-10-27 06:00:00——2023-10-28 00:00:00 的数据
dwd_solar_feature_tmp = dwd_solar_feature_tmp[dwd_solar_feature_tmp["valid_datetime"] >= "2023-10-27 06:00:00"]
dwd_solar_feature_tmp = dwd_solar_feature_tmp[dwd_solar_feature_tmp["valid_datetime"] <= "2023-10-28 00:00:00"]

#3. 删除非最新的天气预报数据
dwd_solar_features = dwd_solar_features[dwd_solar_features["valid_datetime"] - dwd_solar_features["ref_datetime"] < np.timedelta64(6,"h")]

#4. 将最新的天气预报数据与原始数据合并
dwd_solar_features=pd.concat([dwd_solar_features,dwd_solar_feature_tmp],axis=0)

#按照valid_datetime，插值为半小时分辨率的数据
dwd_solar_features = dwd_solar_features.set_index("valid_datetime").resample("30T").interpolate("linear")
dwd_solar_features=dwd_solar_features.reset_index()

#删除最后一条数据
dwd_solar_features = dwd_solar_features.iloc[:-1,:]

#清理临时数据
del dwd_solar_feature_tmp


#%% 发电数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]


#%% 创建风电预测数据集
columns=['valid_datetime','ws_100_t-1','wd_100_t-1','temp_t-1','ws_100_t','wd_100_t','temp_t','ws_100_t+1','wd_100_t+1','temp_t+1','Wind_MWh_credit']
WindDataset=pd.DataFrame(columns=columns)

for i in tqdm(range(len(dwd_Hornsea1_features)-2)):
    WindDataset.loc[i]=[dwd_Hornsea1_features.loc[i+1,'valid_datetime'],
                        dwd_Hornsea1_features.loc[i,'WindSpeed:100'],
                        dwd_Hornsea1_features.loc[i,'WindDirection:100'],
                        dwd_Hornsea1_features.loc[i,'Temperature'],
                        dwd_Hornsea1_features.loc[i+1,'WindSpeed:100'],
                        dwd_Hornsea1_features.loc[i+1,'WindDirection:100'],
                        dwd_Hornsea1_features.loc[i+1,'Temperature'],
                        dwd_Hornsea1_features.loc[i+2,'WindSpeed:100'],
                        dwd_Hornsea1_features.loc[i+2,'WindDirection:100'],
                        dwd_Hornsea1_features.loc[i+2,'Temperature'],
                        energy_data.loc[i+1,'Wind_MWh_credit']]
    

#%% 创建光伏预测数据集
columns=['valid_datetime','rad_t-1','cloud_t-1','rad_t','cloud_t','rad_t+1','cloud_t+1','Solar_MWh_credit']
SolarDataset=pd.DataFrame(columns=columns)

for i in tqdm(range(len(dwd_solar_features)-2)):
    SolarDataset.loc[i]=[dwd_solar_features.loc[i+1,'valid_datetime'],
                        dwd_solar_features.loc[i,'SolarDownwardRadiation'],
                        dwd_solar_features.loc[i,'CloudCover'],
                        dwd_solar_features.loc[i+1,'SolarDownwardRadiation'],
                        dwd_solar_features.loc[i+1,'CloudCover'],
                        dwd_solar_features.loc[i+2,'SolarDownwardRadiation'],
                        dwd_solar_features.loc[i+2,'CloudCover'],
                        energy_data.loc[i+1,'Solar_MWh_credit']]


#%% 将风电和光伏数据集整合
IntegratedDataset=WindDataset.merge(SolarDataset,how='outer',on='valid_datetime')
IntegratedDataset["total_generation_MWh"]=IntegratedDataset["Wind_MWh_credit"]+IntegratedDataset["Solar_MWh_credit"]
IntegratedDataset=IntegratedDataset.drop(columns=['Wind_MWh_credit','Solar_MWh_credit'])


#%% 去掉含有缺失值的条目
SolarDataset=SolarDataset.dropna(axis=0,how='any')
WindDataset=WindDataset.dropna(axis=0,how='any')
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')



#%%  导出数据集
WindDataset.to_csv('data/dataset/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/SolarDataset.csv',index=False)
IntegratedDataset.to_csv('data/dataset/IntegratedDataset.csv',index=False)






















