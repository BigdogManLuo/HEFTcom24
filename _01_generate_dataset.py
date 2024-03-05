import pandas as pd
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

#%%###############################合并Hornsea1_数据############################################
dwd_Hornsea1_old = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_new = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
dwd_Hornsea1_latest = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20240108_20240129.nc")

latitude=[53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
longitude=[1.702, 1.767, 1.832, 1.897, 1.962, 2.027]
DWD_Hornsea1_Features={}

for la in tqdm(latitude):
    for lo in longitude:
        dwd_Hornsea1_features_old=dwd_Hornsea1_old[["WindSpeed:100"]].sel(latitude=la,longitude=lo,method="nearest").to_dataframe().reset_index()
        dwd_Hornsea1_features_new=dwd_Hornsea1_new[["WindSpeed:100"]].sel(latitude=la,longitude=lo,method="nearest").to_dataframe().reset_index()
        dwd_Hornsea1_features_latest=dwd_Hornsea1_latest[["WindSpeed:100"]].sel(latitude=la,longitude=lo,method="nearest").to_dataframe().reset_index()

        dwd_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
        dwd_Hornsea1_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

        #=======================================合并=======================================
        dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
        dwd_Hornsea1_features_new = dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
        dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new,dwd_Hornsea1_features_latest],axis=0).reset_index(drop=True)


        #=======================================标准化时间=======================================
        dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
        dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

        #=======================================删除非最新的天气预报数据=======================================
        dwd_Hornsea1_features = dwd_Hornsea1_features[dwd_Hornsea1_features["valid_datetime"] - dwd_Hornsea1_features["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)

        #===============================================插值=======================================
        dwd_Hornsea1_features=dwd_Hornsea1_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
        dwd_Hornsea1_features = dwd_Hornsea1_features.drop(columns="ref_datetime",axis=1).reset_index()

        DWD_Hornsea1_Features[(la,lo)]=dwd_Hornsea1_features


#%%###############################合并光伏数据############################################
dwd_solar_old=xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_new=xr.open_dataset("data/dwd_icon_eu_pes10_20231027_20240108.nc")
dwd_solar_latest=xr.open_dataset("data/dwd_icon_eu_pes10_20240108_20240129.nc")

points=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
DWD_Solar_Features={}

for point in tqdm(points):
    dwd_solar_features_old=dwd_solar_old[["SolarDownwardRadiation"]].sel(point=point).to_dataframe().reset_index()
    dwd_solar_features_new=dwd_solar_new[["SolarDownwardRadiation"]].sel(point=point).to_dataframe().reset_index()
    dwd_solar_features_latest=dwd_solar_latest[["SolarDownwardRadiation"]].sel(point=point).to_dataframe().reset_index()

    dwd_solar_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
    dwd_solar_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

    #=======================================合并=======================================
    dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    dwd_solar_features_new = dwd_solar_features_new[dwd_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new,dwd_solar_features_latest],axis=0).reset_index(drop=True)

    #=======================================标准化时间=======================================
    dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
    dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

    #=======================================删除非最新的天气预报数据=======================================
    dwd_solar_features = dwd_solar_features[dwd_solar_features["valid_datetime"] - dwd_solar_features["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)

    #===============================================插值=======================================
    dwd_solar_features=dwd_solar_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
    dwd_solar_features = dwd_solar_features.drop(columns="ref_datetime",axis=1).reset_index()

    DWD_Solar_Features[point]=dwd_solar_features


#%% 能源数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20240118.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])

#半小时MWh单位
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]


#%% 整合
IntegratedDataset=pd.DataFrame()

for i in tqdm(range(DWD_Hornsea1_Features[(la,lo)].shape[0])): #遍历所有的数据点
    data={}
    for j, (key, dwd_Hornsea1_features) in enumerate(DWD_Hornsea1_Features.items()):  #遍历每个坐标
        data[f"ws{j}"] = dwd_Hornsea1_features.iloc[i]["WindSpeed:100"]

    for j, (key, dwd_solar_features) in enumerate(DWD_Solar_Features.items()):  #遍历每个坐标
        data[f"rad{j}"] = dwd_solar_features.iloc[i]["SolarDownwardRadiation"]
    
    data["valid_datetime"] = dwd_Hornsea1_features.iloc[i]["valid_datetime"]
    data["ref_datetime"] = dwd_Hornsea1_features.iloc[i]["ref_datetime"]
    
    IntegratedDataset=IntegratedDataset._append(data,ignore_index=True)

IntegratedDataset=IntegratedDataset.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")


#%% 数据清洗
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')
IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<650]

#%% 特征工程

#提取valid_datetime中的小时
IntegratedDataset["hours"]=pd.to_datetime(IntegratedDataset["valid_datetime"]).dt.hour

IntegratedDataset=pd.read_csv('data/dataset/dwd/IntegratedDataset.csv')

#%% 分别构造风电、光伏数据集
#columns_wind_features=[f"ws{i}" for i in range(36)]
columns_wind_features=["ws13","ws14","ws18","ws12","ws24","ws0","ws15","ws25","ws30","ws19","ws35","ws1","ws6","ws5","ws31","ws8","ws32","ws2","ws34","ws7","ws4"]
columns_wind_labels=["Wind_MWh_credit"]
#columns_solar_features=[f"rad{i}" for i in range(20)]
columns_solar_features=["rad12","rad19","rad7","rad8","rad3","rad14","rad1","rad2","rad11","rad0","rad13","rad17"]
columns_solar_labels=["Solar_MWh_credit"]

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
with open('data/dataset/dwd/Dataset_stats.pkl', 'wb') as handle:
    pickle.dump(Dataset_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

#加入不需要标准化的特征
hours = IntegratedDataset["hours"].copy()
SolarDataset.insert(len(SolarDataset.columns) - 1, "hours", hours)

#%% 导出数据集
IntegratedDataset.to_csv('data/dataset/dwd/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/dwd/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/dwd/SolarDataset.csv',index=False)


