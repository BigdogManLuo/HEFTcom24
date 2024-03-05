import pandas as pd
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

#%%===================================================处理dwd 风电数据==========================================================
dwd_Hornsea1_old = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_new = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
dwd_Hornsea1_latest = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20240108_20240129.nc")

#地区平均风速特征
dwd_Hornsea1_features_old=dwd_Hornsea1_old[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()

dwd_Hornsea1_features_new=dwd_Hornsea1_new[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

dwd_Hornsea1_features_latest=dwd_Hornsea1_latest[["WindSpeed:100","WindDirection:100"]].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

#地区最大风速特征
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(
    dwd_Hornsea1_old["WindSpeed:100"].max(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"maxWindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"]) 

dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(
    dwd_Hornsea1_new["WindSpeed:100"].max(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"maxWindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_latest=dwd_Hornsea1_features_latest.merge(
    dwd_Hornsea1_latest["WindSpeed:100"].max(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"maxWindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"]) 

#地区最小风速特征
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(
    dwd_Hornsea1_old["WindSpeed:100"].min(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"minWindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"]) 

dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(
    dwd_Hornsea1_new["WindSpeed:100"].min(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"minWindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_latest=dwd_Hornsea1_features_latest.merge(
    dwd_Hornsea1_latest["WindSpeed:100"].min(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"minWindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#地区75%分位数风速特征
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(
    dwd_Hornsea1_old["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.75).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q75WindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(
    dwd_Hornsea1_new["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.75).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q75WindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_latest=dwd_Hornsea1_features_latest.merge(
    dwd_Hornsea1_latest["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.75).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q75WindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#地区25%分位数风速特征
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(
    dwd_Hornsea1_old["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.25).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q25WindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(
    dwd_Hornsea1_new["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.25).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q25WindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_latest=dwd_Hornsea1_features_latest.merge(
    dwd_Hornsea1_latest["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.25).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q25WindSpeed:100","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#合并
dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
dwd_Hornsea1_features_new = dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new,dwd_Hornsea1_features_latest],axis=0).reset_index(drop=True)


#标准化时间
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

#删除非最新的天气预报数据
dwd_Hornsea1_features = dwd_Hornsea1_features[dwd_Hornsea1_features["valid_datetime"] - dwd_Hornsea1_features["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)

#插值
dwd_Hornsea1_features=dwd_Hornsea1_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
dwd_Hornsea1_features = dwd_Hornsea1_features.drop(columns="ref_datetime",axis=1).reset_index()


#%%===================================================处理光伏数据================================================
dwd_solar_old=xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_new=xr.open_dataset("data/dwd_icon_eu_pes10_20231027_20240108.nc")
dwd_solar_latest=xr.open_dataset("data/dwd_icon_eu_pes10_20240108_20240129.nc")

#地区平均辐照度特征
dwd_solar_features_old=dwd_solar_old["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()

dwd_solar_features_new=dwd_solar_new["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_new=dwd_solar_features_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"})

dwd_solar_features_latest=dwd_solar_latest["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_latest=dwd_solar_features_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"})

#地区最大辐照度特征
dwd_solar_features_old=dwd_solar_features_old.merge(
    dwd_solar_old["SolarDownwardRadiation"].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_features_new.merge(
    dwd_solar_new["SolarDownwardRadiation"].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_latest=dwd_solar_features_latest.merge(
    dwd_solar_latest["SolarDownwardRadiation"].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#地区最小辐照度特征
dwd_solar_features_old=dwd_solar_features_old.merge(
    dwd_solar_old["SolarDownwardRadiation"].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_features_new.merge(
    dwd_solar_new["SolarDownwardRadiation"].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_latest=dwd_solar_features_latest.merge(
    dwd_solar_latest["SolarDownwardRadiation"].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#地区75%分位数辐照度特征
dwd_solar_features_old=dwd_solar_features_old.merge(
    dwd_solar_old["SolarDownwardRadiation"].quantile(dim="point",q=0.75).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q75SolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_features_new.merge(
    dwd_solar_new["SolarDownwardRadiation"].quantile(dim="point",q=0.75).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q75SolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_latest=dwd_solar_features_latest.merge(
    dwd_solar_latest["SolarDownwardRadiation"].quantile(dim="point",q=0.75).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q75SolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#地区25%分位数辐照度特征
dwd_solar_features_old=dwd_solar_features_old.merge(
    dwd_solar_old["SolarDownwardRadiation"].quantile(dim="point",q=0.25).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q25SolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_features_new.merge(
    dwd_solar_new["SolarDownwardRadiation"].quantile(dim="point",q=0.25).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q25SolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

dwd_solar_features_latest=dwd_solar_features_latest.merge(
    dwd_solar_latest["SolarDownwardRadiation"].quantile(dim="point",q=0.25).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q25SolarDownwardRadiation","reference_time":"ref_datetime","valid_time":"valid_datetime"}),
    how="left",on=["ref_datetime","valid_datetime"])

#合并
dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
dwd_solar_features_new = dwd_solar_features_new[dwd_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new,dwd_solar_features_latest],axis=0).reset_index(drop=True)

#标准化时间
dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

#删除非最新的天气预报数据
dwd_solar_features = dwd_solar_features[dwd_solar_features["valid_datetime"] - dwd_solar_features["ref_datetime"] < np.timedelta64(48,"h")]

#插值
dwd_solar_features=dwd_solar_features.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
dwd_solar_features = dwd_solar_features.drop(columns="ref_datetime",axis=1).reset_index()


#%%================================================能源数据======================================================
energy_data = pd.read_csv("data/Energy_Data_20200920_20240118.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])

#半小时MWh单位
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]

#缩减为时间-风电-光伏-光伏容量 四列
energy_data = energy_data[["dtm","Wind_MWh_credit","Solar_MWh_credit","Solar_capacity_mwp"]]

#%%=================================================整合==========================================================

#合并dwd数据
modelling_table=dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
modelling_table= modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")

# 按照"ref_datetime"分组
modelling_table=modelling_table.groupby("ref_datetime")
IntegratedDataset=pd.DataFrame()

#遍历每组group
for group_idx,(ref_datetime,group) in enumerate(tqdm(modelling_table)):
    
    for i in range(len(group)-2):
        data={
            
            "ws_100_t-1_dwd_1":group.iloc[i]["WindSpeed:100"],
            "ws_100_t-1_dwd_max":group.iloc[i]["maxWindSpeed:100"],
            "ws_100_t-1_dwd_min":group.iloc[i]["minWindSpeed:100"],
            "ws_100_t-1_dwd_q75":group.iloc[i]["q75WindSpeed:100"],
            "ws_100_t-1_dwd_q25":group.iloc[i]["q25WindSpeed:100"],
            
            "ws_100_t_dwd_1":group.iloc[i+1]["WindSpeed:100"],
            "ws_100_t_dwd_max":group.iloc[i+1]["maxWindSpeed:100"],
            "ws_100_t_dwd_min":group.iloc[i+1]["minWindSpeed:100"],
            "ws_100_t_dwd_q75":group.iloc[i+1]["q75WindSpeed:100"],
            "ws_100_t_dwd_q25":group.iloc[i+1]["q25WindSpeed:100"],
            
            "ws_100_t+1_dwd_1":group.iloc[i+2]["WindSpeed:100"],
            "ws_100_t+1_dwd_max":group.iloc[i+2]["maxWindSpeed:100"],
            "ws_100_t+1_dwd_min":group.iloc[i+2]["minWindSpeed:100"],
            "ws_100_t+1_dwd_q75":group.iloc[i+2]["q75WindSpeed:100"],
            "ws_100_t+1_dwd_q25":group.iloc[i+2]["q25WindSpeed:100"],
            
            "rad_t-1_dwd":group.iloc[i]["SolarDownwardRadiation"],
            "rad_t-1_dwd_max":group.iloc[i]["maxSolarDownwardRadiation"],
            "rad_t-1_dwd_min":group.iloc[i]["minSolarDownwardRadiation"],
            "rad_t-1_dwd_q75":group.iloc[i]["q75SolarDownwardRadiation"],
            "rad_t-1_dwd_q25":group.iloc[i]["q25SolarDownwardRadiation"],            
            
            "rad_t_dwd":group.iloc[i+1]["SolarDownwardRadiation"],
            "rad_t_dwd_max":group.iloc[i+1]["maxSolarDownwardRadiation"],
            "rad_t_dwd_min":group.iloc[i+1]["minSolarDownwardRadiation"],
            "rad_t_dwd_q75":group.iloc[i+1]["q75SolarDownwardRadiation"],
            "rad_t_dwd_q25":group.iloc[i+1]["q25SolarDownwardRadiation"],
            
            "rad_t+1_dwd":group.iloc[i+2]["SolarDownwardRadiation"],
            "rad_t+1_dwd_max":group.iloc[i+2]["maxSolarDownwardRadiation"],
            "rad_t+1_dwd_min":group.iloc[i+2]["minSolarDownwardRadiation"],
            "rad_t+1_dwd_q75":group.iloc[i+2]["q75SolarDownwardRadiation"],
            "rad_t+1_dwd_q25":group.iloc[i+2]["q25SolarDownwardRadiation"],
            
            "ref_datetime":group.iloc[i+1]["ref_datetime"],
            "valid_datetime":group.iloc[i+1]["valid_datetime"],
            "Wind_MWh_credit":group.iloc[i+1]["Wind_MWh_credit"],
            "Solar_MWh_credit":group.iloc[i+1]["Solar_MWh_credit"],
            "Solar_capacity_mwp":group.iloc[i+1]["Solar_capacity_mwp"],
            "total_generation_MWh":group.iloc[i+1]["Wind_MWh_credit"] + group.iloc[i+1]["Solar_MWh_credit"]
            }
        IntegratedDataset=IntegratedDataset._append(data,ignore_index=True)
        
#%%============================================数据清洗======================================================

#缺失值处理
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')

#删除Wind_MWh_credit中超过650的异常值
IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<650]

#%%============================================特征工程======================================================

#提取valid_datetime中的小时
IntegratedDataset["hours"]=pd.to_datetime(IntegratedDataset["valid_datetime"]).dt.hour

#%%============================================分别构造风电、光伏数据集========================================

#IntegratedDataset=pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")

columns_wind_features=["ws_100_t_dwd_1","ws_100_t_dwd_max","ws_100_t_dwd_min","ws_100_t_dwd_q75","ws_100_t_dwd_q25",
                            "ws_100_t+1_dwd_1","ws_100_t+1_dwd_max","ws_100_t+1_dwd_min","ws_100_t+1_dwd_q75","ws_100_t+1_dwd_q25"]
columns_wind_labels=["Wind_MWh_credit"]
columns_solar_features=["rad_t-1_dwd","rad_t-1_dwd_max","rad_t-1_dwd_min","rad_t-1_dwd_q75","rad_t-1_dwd_q25",
                        "rad_t_dwd","rad_t_dwd_max","rad_t_dwd_min","rad_t_dwd_q75","rad_t_dwd_q25"]
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

#%%==========================================导出数据集========================================
IntegratedDataset.to_csv('data/dataset/dwd/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/dwd/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/dwd/SolarDataset.csv',index=False)

#%%==========================================可视化========================================

#展示功率-天气相关性
plt.figure(figsize=(9,5))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
plt.subplot(3,1,1)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t-1_dwd_1", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('ws_100_t_dwd_1')
plt.ylabel('Generation [MWh]')
plt.subplot(3,1,2)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t_dwd_max", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel("ws_100_t_dwd_max")
plt.ylabel('Generation [MWh]')
plt.subplot(3,1,3)
sns.scatterplot(data=IntegratedDataset, x="ws_100_t+1_dwd_min", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel("ws_100_t+1_dwd_min",)
plt.ylabel('Generation [MWh]')
plt.tight_layout()


plt.figure(figsize=(9,5))
plt.subplot(1,3,1)
sns.scatterplot(data=IntegratedDataset, x="rad_t_dwd", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t_dwd')
plt.ylabel('Generation [MWh]')
plt.subplot(1,3,2)
sns.scatterplot(data=IntegratedDataset, x="rad_t-1_dwd", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t-1_dwd')
plt.ylabel('Generation [MWh]')
plt.subplot(1,3,3)
sns.scatterplot(data=IntegratedDataset, x="rad_t+1_dwd", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('rad_t+1_dwd')
plt.ylabel('Generation [MWh]')
plt.tight_layout()

