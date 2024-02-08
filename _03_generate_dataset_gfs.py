import pandas as pd
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

#%% 处理gfs 风电数据
gfs_Hornsea1_old = xr.open_dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
gfs_Hornsea1_new = xr.open_dataset("data/ncep_gfs_hornsea_1_20231027_20240108.nc")
gfs_Hornsea1_latest = xr.open_dataset("data/ncep_gfs_hornsea_1_20240108_20240129.nc")

#特征提取
gfs_Hornsea1_features_old=gfs_Hornsea1_old[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
gfs_Hornsea1_features_new=gfs_Hornsea1_new[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
gfs_Hornsea1_features_latest=gfs_Hornsea1_latest[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()

#合并
gfs_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
gfs_Hornsea1_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

gfs_Hornsea1_features_old = gfs_Hornsea1_features_old[gfs_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
gfs_Hornsea1_features_new = gfs_Hornsea1_features_new[gfs_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
gfs_Hornsea1_features=pd.concat([gfs_Hornsea1_features_old,gfs_Hornsea1_features_new,gfs_Hornsea1_features_latest],axis=0).reset_index(drop=True)


#标准化时间
gfs_Hornsea1_features["ref_datetime"] = gfs_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
gfs_Hornsea1_features["valid_datetime"] = gfs_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(gfs_Hornsea1_features["valid_datetime"],unit="hours")


gfs_Hornsea1_features_latest["ref_datetime"] = gfs_Hornsea1_features_latest["ref_datetime"].dt.tz_localize("UTC")
gfs_Hornsea1_features_latest["valid_datetime"] = gfs_Hornsea1_features_latest["ref_datetime"] + pd.TimedeltaIndex(gfs_Hornsea1_features_latest["valid_datetime"],unit="hours")



#删除非最新的天气预报数据
gfs_Hornsea1_features = gfs_Hornsea1_features[gfs_Hornsea1_features["valid_datetime"] - gfs_Hornsea1_features["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)

#插值
gfs_Hornsea1_features=gfs_Hornsea1_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
gfs_Hornsea1_features = gfs_Hornsea1_features.drop(columns="ref_datetime",axis=1).reset_index()

#%% 处理gfs 光伏数据
gfs_solar_old=xr.open_dataset("data/ncep_gfs_pes10_20200920_20231027.nc")
gfs_solar_new=xr.open_dataset("data/ncep_gfs_pes10_20231027_20240108.nc")
gfs_solar_latest=xr.open_dataset("data/ncep_gfs_pes10_20240108_20240129.nc")

#特征提取
gfs_solar_features_old=gfs_solar_old["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
gfs_solar_features_new=gfs_solar_new["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
gfs_solar_features_latest=gfs_solar_latest["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()

#合并
gfs_solar_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
gfs_solar_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

gfs_solar_features_old = gfs_solar_features_old[gfs_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
gfs_solar_features_new = gfs_solar_features_new[gfs_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
gfs_solar_features=pd.concat([gfs_solar_features_old,gfs_solar_features_new,gfs_solar_features_latest],axis=0).reset_index(drop=True)


#标准化时间
gfs_solar_features["ref_datetime"] = gfs_solar_features["ref_datetime"].dt.tz_localize("UTC")
gfs_solar_features["valid_datetime"] = gfs_solar_features["ref_datetime"] + pd.TimedeltaIndex(gfs_solar_features["valid_datetime"],unit="hours")


#删除非最新的天气预报数据
gfs_solar_features = gfs_solar_features[gfs_solar_features["valid_datetime"] - gfs_solar_features["ref_datetime"] < np.timedelta64(48,"h")]

#插值
gfs_solar_features=gfs_solar_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
gfs_solar_features = gfs_solar_features.drop(columns="ref_datetime",axis=1).reset_index()


#%% 能源数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20240118.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])

energy_data_latest = pd.read_csv("data/Energy_Data_latest.csv")
energy_data_latest["dtm"] = pd.to_datetime(energy_data_latest["dtm"])
energy_data_latest.rename(columns={"capacity_mwp":"Solar_capacity_mwp"},inplace=True)

#半小时MWh单位
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]

energy_data_latest["Wind_MWh_credit"] = 0.5*energy_data_latest["Wind_MW"] - energy_data_latest["boa"]
energy_data_latest["Solar_MWh_credit"] = 0.5*energy_data_latest["Solar_MW"]

#缩减为时间-风电-光伏-光伏容量 四列
energy_data = energy_data[["dtm","Wind_MWh_credit","Solar_MWh_credit","Solar_capacity_mwp"]]
energy_data_latest=energy_data_latest[["dtm","Wind_MWh_credit","Solar_MWh_credit","Solar_capacity_mwp"]]

#合并，有重复的条目则覆盖
energy_data = pd.concat([energy_data,energy_data_latest],axis=0).drop_duplicates(subset=["dtm"],keep="last").reset_index(drop=True)

#%% 整合
#合并gfs数据
modelling_table=gfs_Hornsea1_features.merge(gfs_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
modelling_table= modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")

# 按照"ref_datetime"分组
modelling_table=modelling_table.groupby("ref_datetime")
IntegratedDataset=pd.DataFrame()

#遍历每组group
for group_idx,(ref_datetime,group) in enumerate(tqdm(modelling_table)):
    
    for i in range(len(group)-2):
        data={
            
            "ref_datetime":group.iloc[i+1]["ref_datetime"],
            "valid_datetime":group.iloc[i+1]["valid_datetime"],

            "ws_100_t-1_gfs_1":group.iloc[i]["WindSpeed:100"],
            "wd_100_t-1_gfs":group.iloc[i]["WindDirection:100"],
            "ws_100_t_gfs_1":group.iloc[i+1]["WindSpeed:100"],
            "wd_100_t_gfs":group.iloc[i+1]["WindDirection:100"],
            "ws_100_t+1_gfs_1":group.iloc[i+2]["WindSpeed:100"],
            "wd_100_t+1_gfs":group.iloc[i+2]["WindDirection:100"],

            "rad_t-1_gfs":group.iloc[i]["SolarDownwardRadiation"],
            "rad_t_gfs":group.iloc[i+1]["SolarDownwardRadiation"],
            "rad_t+1_gfs":group.iloc[i+2]["SolarDownwardRadiation"],
            
            "Wind_MWh_credit":group.iloc[i+1]["Wind_MWh_credit"],
            "Solar_MWh_credit":group.iloc[i+1]["Solar_MWh_credit"],
            "Solar_capacity_mwp":group.iloc[i+1]["Solar_capacity_mwp"],
            "total_generation_MWh":group.iloc[i+1]["Wind_MWh_credit"] + group.iloc[i+1]["Solar_MWh_credit"]
            }
        IntegratedDataset=IntegratedDataset._append(data,ignore_index=True)
#%% 数据清洗

#缺失值处理
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')

#删除Wind_MWh_credit中超过650的异常值
IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<650]

#%% 特征工程

#将风向角度变为正余弦值
IntegratedDataset["wd_100_t-1_gfs_cos"]=np.cos(np.radians(IntegratedDataset["wd_100_t-1_gfs"]))
IntegratedDataset["wd_100_t_gfs_cos"]=np.cos(np.radians(IntegratedDataset["wd_100_t_gfs"]))
IntegratedDataset["wd_100_t+1_gfs_cos"]=np.cos(np.radians(IntegratedDataset["wd_100_t+1_gfs"]))
IntegratedDataset["wd_100_t-1_gfs_sin"]=np.sin(np.radians(IntegratedDataset["wd_100_t-1_gfs"]))
IntegratedDataset["wd_100_t_gfs_sin"]=np.sin(np.radians(IntegratedDataset["wd_100_t_gfs"]))
IntegratedDataset["wd_100_t+1_gfs_sin"]=np.sin(np.radians(IntegratedDataset["wd_100_t+1_gfs"]))

#将风速变为二次方和三次方
IntegratedDataset["ws_100_t-1_gfs_2"]=IntegratedDataset["ws_100_t-1_gfs_1"]**2
IntegratedDataset["ws_100_t_gfs_2"]=IntegratedDataset["ws_100_t_gfs_1"]**2
IntegratedDataset["ws_100_t+1_gfs_2"]=IntegratedDataset["ws_100_t+1_gfs_1"]**2
IntegratedDataset["ws_100_t-1_gfs_3"]=IntegratedDataset["ws_100_t-1_gfs_1"]**3
IntegratedDataset["ws_100_t_gfs_3"]=IntegratedDataset["ws_100_t_gfs_1"]**3
IntegratedDataset["ws_100_t+1_gfs_3"]=IntegratedDataset["ws_100_t+1_gfs_1"]**3

#提取valid_datetime中的小时
IntegratedDataset["hours"]=pd.to_datetime(IntegratedDataset["valid_datetime"]).dt.hour

#提取valid_datetime-ref_datetime的小时差,以2小时为间隔
IntegratedDataset["hours_diff"]=pd.to_datetime(IntegratedDataset["valid_datetime"])-pd.to_datetime(IntegratedDataset["ref_datetime"])
IntegratedDataset["hours_diff"]=IntegratedDataset["hours_diff"].dt.total_seconds()/3600
IntegratedDataset["hours_diff"]=IntegratedDataset["hours_diff"].astype(int)
IntegratedDataset["hours_diff"]=(IntegratedDataset["hours_diff"]/2).astype(int)

#IntegratedDataset=pd.read_csv("data/dataset/gfs/IntegratedDataset.csv")

#%% 分别构造风电、光伏数据集
columns_wind_features=["ws_100_t-1_gfs_1",
                       "ws_100_t_gfs_1",
                       "ws_100_t+1_gfs_1"]
columns_wind_labels=["Wind_MWh_credit"]
columns_solar_features=["rad_t-1_gfs","rad_t_gfs","rad_t+1_gfs"]
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
with open('data/dataset/gfs/Dataset_stats.pkl', 'wb') as handle:
    pickle.dump(Dataset_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

#加入不需要标准化的特征
hours = IntegratedDataset["hours"].copy()
SolarDataset.insert(len(SolarDataset.columns) - 1, "hours", hours)


#%% 导出数据集
IntegratedDataset.to_csv('data/dataset/gfs/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/gfs/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/gfs/SolarDataset.csv',index=False)

#%% 可视化

#展示功率-天气相关性
plt.figure(figsize=(9,5))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.subplot(2,2,1)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t_gfs_1", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('ws_100_t_gfs_1')
plt.ylabel('Generation [MWh]')
plt.subplot(2,2,2)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t+1_gfs_1", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('ws_100_t+1_gfs_1')
plt.ylabel('Generation [MWh]')
plt.subplot(2,2,3)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t+1_gfs_2", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('ws_100_t+1_gfs_2')
plt.ylabel('Generation [MWh]')
plt.subplot(2,2,4)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t+1_gfs_3", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('ws_100_t+1_gfs_3')
plt.ylabel('Generation [MWh]')
plt.tight_layout()



plt.figure(figsize=(9,5))
plt.subplot(1,3,1)
sns.scatterplot(data=IntegratedDataset, x="rad_t_gfs", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t_gfs')
plt.ylabel('Generation [MWh]')
plt.subplot(1,3,2)
sns.scatterplot(data=IntegratedDataset, x="rad_t-1_gfs", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t-1_gfs')
plt.ylabel('Generation [MWh]')
plt.subplot(1,3,3)
sns.scatterplot(data=IntegratedDataset, x="rad_t+1_gfs", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t+1_gfs')
plt.ylabel('Generation [MWh]')
plt.tight_layout()

