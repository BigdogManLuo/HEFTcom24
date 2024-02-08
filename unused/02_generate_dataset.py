import pandas as pd
import xarray as xr
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
import seaborn as sns

#%%清空路径
'''
for root, dirs, files in os.walk("data/dataset"):
    for name in files:
        os.remove(os.path.join(root, name))
'''

#%% 处理dwd 风电数据
dwd_Hornsea1_old = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_new = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20231027_20240108.nc")

#特征提取
dwd_Hornsea1_features_old=dwd_Hornsea1_old["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(dwd_Hornsea1_old["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(dwd_Hornsea1_old["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_new=dwd_Hornsea1_new["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(dwd_Hornsea1_new["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(dwd_Hornsea1_new["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])

#标准化时间
dwd_Hornsea1_features_old["ref_datetime"] = dwd_Hornsea1_features_old["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features_old["valid_datetime"] = dwd_Hornsea1_features_old["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features_old["valid_datetime"],unit="hours")

dwd_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
dwd_Hornsea1_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
dwd_Hornsea1_features_new["ref_datetime"] = dwd_Hornsea1_features_new["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features_new["valid_datetime"] = dwd_Hornsea1_features_new["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features_new["valid_datetime"],unit="hours")


#删除非最新的天气预报数据
dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["valid_datetime"] - dwd_Hornsea1_features_old["ref_datetime"] < np.timedelta64(6,"h")]
dwd_Hornsea1_features_new = dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["valid_datetime"] - dwd_Hornsea1_features_new["ref_datetime"] < np.timedelta64(6,"h")]

#合并新旧数据
dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new],axis=0).reset_index(drop=True)

#按照valid_datetime，插值为半小时分辨率的数据
dwd_Hornsea1_features = dwd_Hornsea1_features.set_index("valid_datetime").resample("30T").interpolate("linear")
dwd_Hornsea1_features=dwd_Hornsea1_features.reset_index()

#将风向角度变为弧度制
dwd_Hornsea1_features["WindDirection:100_rad"]=np.radians(dwd_Hornsea1_features["WindDirection:100"])

#将风速变为二次方和三次方
dwd_Hornsea1_features["WindSpeed:100^2"]=dwd_Hornsea1_features["WindSpeed:100"]**2
dwd_Hornsea1_features["WindSpeed:100^3"]=dwd_Hornsea1_features["WindSpeed:100"]**3

#%% 处理gfs 风电数据
gfs_Hornsea1_old = xr.open_dataset("data/ncep_gfs_hornsea_1_20200920_20231027.nc")
gfs_Hornsea1_new = xr.open_dataset("data/ncep_gfs_hornsea_1_20231027_20240108.nc")

#特征提取
gfs_Hornsea1_features_old=gfs_Hornsea1_old["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
gfs_Hornsea1_features_old=gfs_Hornsea1_features_old.merge(gfs_Hornsea1_old["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
gfs_Hornsea1_features_old=gfs_Hornsea1_features_old.merge(gfs_Hornsea1_old["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

gfs_Hornsea1_features_new=gfs_Hornsea1_new["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
gfs_Hornsea1_features_new=gfs_Hornsea1_features_new.merge(gfs_Hornsea1_new["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])
gfs_Hornsea1_features_new=gfs_Hornsea1_features_new.merge(gfs_Hornsea1_new["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])

#标准化时间
gfs_Hornsea1_features_old["ref_datetime"] = gfs_Hornsea1_features_old["ref_datetime"].dt.tz_localize("UTC")
gfs_Hornsea1_features_old["valid_datetime"] = gfs_Hornsea1_features_old["ref_datetime"] + pd.TimedeltaIndex(gfs_Hornsea1_features_old["valid_datetime"],unit="hours")

gfs_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
gfs_Hornsea1_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
gfs_Hornsea1_features_new["ref_datetime"] = gfs_Hornsea1_features_new["ref_datetime"].dt.tz_localize("UTC")
gfs_Hornsea1_features_new["valid_datetime"] = gfs_Hornsea1_features_new["ref_datetime"] + pd.TimedeltaIndex(gfs_Hornsea1_features_new["valid_datetime"],unit="hours")

#删除非最新的天气预报数据
gfs_Hornsea1_features_old = gfs_Hornsea1_features_old[gfs_Hornsea1_features_old["valid_datetime"] - gfs_Hornsea1_features_old["ref_datetime"] < np.timedelta64(6,"h")]

gfs_Hornsea1_features_new = gfs_Hornsea1_features_new[gfs_Hornsea1_features_new["valid_datetime"] - gfs_Hornsea1_features_new["ref_datetime"] < np.timedelta64(6,"h")]

#合并新旧数据
gfs_Hornsea1_features_old = gfs_Hornsea1_features_old[gfs_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
gfs_Hornsea1_features=pd.concat([gfs_Hornsea1_features_old,gfs_Hornsea1_features_new],axis=0).reset_index(drop=True)

#按照valid_datetime，插值为半小时分辨率的数据
gfs_Hornsea1_features = gfs_Hornsea1_features.set_index("valid_datetime").resample("30T").interpolate("linear")
gfs_Hornsea1_features=gfs_Hornsea1_features.reset_index()

#将风向角度变为弧度制
gfs_Hornsea1_features["WindDirection:100_rad"]=np.radians(gfs_Hornsea1_features["WindDirection:100"])

#将风速变为二次方和三次方
gfs_Hornsea1_features["WindSpeed:100^2"]=gfs_Hornsea1_features["WindSpeed:100"]**2
gfs_Hornsea1_features["WindSpeed:100^3"]=gfs_Hornsea1_features["WindSpeed:100"]**3

#%% 处理dwd 光伏数据
dwd_solar_old=xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_new=xr.open_dataset("data/dwd_icon_eu_pes10_20231027_20240108.nc")

#特征提取
dwd_solar_features_old=dwd_solar_old["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_old=dwd_solar_features_old.merge(dwd_solar_old["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_new["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_new=dwd_solar_features_new.merge(dwd_solar_new["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])

#标准化时间
dwd_solar_features_old["ref_datetime"] = dwd_solar_features_old["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features_old["valid_datetime"] = dwd_solar_features_old["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features_old["valid_datetime"],unit="hours")

dwd_solar_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
dwd_solar_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
dwd_solar_features_new["ref_datetime"] = dwd_solar_features_new["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features_new["valid_datetime"] = dwd_solar_features_new["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features_new["valid_datetime"],unit="hours")

#删除非最新的天气预报数据
dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["valid_datetime"] - dwd_solar_features_old["ref_datetime"] < np.timedelta64(6,"h")]

dwd_solar_features_new = dwd_solar_features_new[dwd_solar_features_new["valid_datetime"] - dwd_solar_features_new["ref_datetime"] < np.timedelta64(6,"h")]

#合并新旧数据
dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new],axis=0).reset_index(drop=True)

#按照valid_datetime，插值为半小时分辨率的数据
dwd_solar_features = dwd_solar_features.set_index("valid_datetime").resample("30T").interpolate("linear")
dwd_solar_features=dwd_solar_features.reset_index()

#%% 处理gfs 光伏数据
gfs_solar_old=xr.open_dataset("data/ncep_gfs_pes10_20200920_20231027.nc")
gfs_solar_new=xr.open_dataset("data/ncep_gfs_pes10_20231027_20240108.nc")

#特征提取
gfs_solar_features_old=gfs_solar_old["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
gfs_solar_features_old=gfs_solar_features_old.merge(gfs_solar_old["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

gfs_solar_features_new=gfs_solar_new["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
gfs_solar_features_new=gfs_solar_features_new.merge(gfs_solar_new["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])

#标准化时间
gfs_solar_features_old["ref_datetime"] = gfs_solar_features_old["ref_datetime"].dt.tz_localize("UTC")
gfs_solar_features_old["valid_datetime"] = gfs_solar_features_old["ref_datetime"] + pd.TimedeltaIndex(gfs_solar_features_old["valid_datetime"],unit="hours")

gfs_solar_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
gfs_solar_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
gfs_solar_features_new["ref_datetime"] = gfs_solar_features_new["ref_datetime"].dt.tz_localize("UTC")
gfs_solar_features_new["valid_datetime"] = gfs_solar_features_new["ref_datetime"] + pd.TimedeltaIndex(gfs_solar_features_new["valid_datetime"],unit="hours")

#删除非最新的天气预报数据
gfs_solar_features_old = gfs_solar_features_old[gfs_solar_features_old["valid_datetime"] - gfs_solar_features_old["ref_datetime"] < np.timedelta64(6,"h")]

gfs_solar_features_new = gfs_solar_features_new[gfs_solar_features_new["valid_datetime"] - gfs_solar_features_new["ref_datetime"] < np.timedelta64(6,"h")]

#合并新旧数据
gfs_solar_features_old = gfs_solar_features_old[gfs_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
gfs_solar_features=pd.concat([gfs_solar_features_old,gfs_solar_features_new],axis=0).reset_index(drop=True)

#按照valid_datetime，插值为半小时分辨率的数据
gfs_solar_features = gfs_solar_features.set_index("valid_datetime").resample("30T").interpolate("linear")
gfs_solar_features=gfs_solar_features.reset_index()


#%% 能源数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data_latest = pd.read_csv("data/energy_data_latest.csv")

#取2023-10-28 00:00 - 2024-01-08 05:00 时间段的数据，与天气预报数据对齐
energy_data_latest = energy_data_latest[energy_data_latest["dtm"] >= "2023-10-28 00:00"]
energy_data_latest = energy_data_latest[energy_data_latest["dtm"] < "2024-01-08 05:30"]

#半小时MWh单位
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]
energy_data_latest["Wind_MWh_credit"] = 0.5*energy_data_latest["Wind_MW"] - energy_data_latest["boa"]
energy_data_latest["Solar_MWh_credit"] = 0.5*energy_data_latest["Solar_MW"]

#缩减为时间-风电-光伏-光伏容量 四列
energy_data = energy_data[["dtm","Wind_MWh_credit","Solar_MWh_credit","Solar_capacity_mwp"]]
energy_data_latest = energy_data_latest[["dtm","Wind_MWh_credit","Solar_MWh_credit","capacity_mwp"]]
energy_data_latest.rename(columns={"capacity_mwp":"Solar_capacity_mwp"},inplace=True)

#合并能源数据
energy_data = pd.concat([energy_data,energy_data_latest],axis=0).reset_index(drop=True)


#%% 创建数据集
IntegratedDataset=pd.DataFrame()

for i in tqdm(range(len(dwd_Hornsea1_features)-3)):

    data={

        "valid_datetime":dwd_Hornsea1_features.loc[i+1,"valid_datetime"],

        "ws_100_t-1_dwd_1":dwd_Hornsea1_features.loc[i,"WindSpeed:100"],
        "ws_100_t-1_dwd_2":dwd_Hornsea1_features.loc[i,"WindSpeed:100^2"],
        "ws_100_t-1_dwd_3":dwd_Hornsea1_features.loc[i,"WindSpeed:100^3"],
        "wd_100_t-1_dwd_cos":np.cos(dwd_Hornsea1_features.loc[i,"WindDirection:100_rad"]),
        "wd_100_t-1_dwd_sin":np.sin(dwd_Hornsea1_features.loc[i,"WindDirection:100_rad"]),
        "temp_t-1_dwd":dwd_Hornsea1_features.loc[i,"Temperature"],
        
        "ws_100_t_dwd_1":dwd_Hornsea1_features.loc[i+1,"WindSpeed:100"],
        "ws_100_t_dwd_2":dwd_Hornsea1_features.loc[i+1,"WindSpeed:100^2"],
        "ws_100_t_dwd_3":dwd_Hornsea1_features.loc[i+1,"WindSpeed:100^3"],
        "wd_100_t_dwd_cos":np.cos(dwd_Hornsea1_features.loc[i+1,"WindDirection:100_rad"]),
        "wd_100_t_dwd_sin":np.sin(dwd_Hornsea1_features.loc[i+1,"WindDirection:100_rad"]),
        "temp_t_dwd":dwd_Hornsea1_features.loc[i+1,"Temperature"],

        "ws_100_t+1_dwd_1":dwd_Hornsea1_features.loc[i+2,"WindSpeed:100"],
        "ws_100_t+1_dwd_2":dwd_Hornsea1_features.loc[i+2,"WindSpeed:100^2"],
        "ws_100_t+1_dwd_3":dwd_Hornsea1_features.loc[i+2,"WindSpeed:100^3"],
        "wd_100_t+1_dwd_cos":np.cos(dwd_Hornsea1_features.loc[i+2,"WindDirection:100_rad"]),
        "wd_100_t+1_dwd_sin":np.sin(dwd_Hornsea1_features.loc[i+2,"WindDirection:100_rad"]),
        "temp_t+1_dwd":dwd_Hornsea1_features.loc[i+2,"Temperature"],
        
        "ws_100_t+2_dwd_1":dwd_Hornsea1_features.loc[i+3,"WindSpeed:100"],
        "ws_100_t+2_dwd_2":dwd_Hornsea1_features.loc[i+3,"WindSpeed:100^2"],
        "ws_100_t+2_dwd_3":dwd_Hornsea1_features.loc[i+3,"WindSpeed:100^3"],
        "wd_100_t+2_dwd_cos":np.cos(dwd_Hornsea1_features.loc[i+3,"WindDirection:100_rad"]),
        "wd_100_t+2_dwd_sin":np.sin(dwd_Hornsea1_features.loc[i+3,"WindDirection:100_rad"]),
        "temp_t+2_dwd":dwd_Hornsea1_features.loc[i+3,"Temperature"],

        "rad_t-1_dwd":dwd_solar_features.loc[i,"SolarDownwardRadiation"],
        "cloud_t-1_dwd":dwd_solar_features.loc[i,"CloudCover"],

        "rad_t_dwd":dwd_solar_features.loc[i+1,"SolarDownwardRadiation"],
        "cloud_t_dwd":dwd_solar_features.loc[i+1,"CloudCover"],

        "rad_t+1_dwd":dwd_solar_features.loc[i+2,"SolarDownwardRadiation"],
        "cloud_t+1_dwd":dwd_solar_features.loc[i+2,"CloudCover"],

        "Wind_MWh_credit":energy_data.loc[i+1,"Wind_MWh_credit"],
        "Solar_MWh_credit":energy_data.loc[i+1,"Solar_MWh_credit"], 
        "Solar_capacity_mwp":energy_data.loc[i+1,"Solar_capacity_mwp"],
        "total_generation_MWh":energy_data.loc[i+1,"Wind_MWh_credit"]+energy_data.loc[i+1,"Solar_MWh_credit"],
    }

    IntegratedDataset=IntegratedDataset._append(data,ignore_index=True)


#缺失值处理
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')

#删除Wind_MWh_credit中超过650的异常值
IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<650]

#%% 风电和光伏预测分别所需的列名
columns_wind_features=["ws_100_t_dwd_1","ws_100_t+1_dwd_1","ws_100_t+1_dwd_2","ws_100_t+1_dwd_3"]
columns_wind_labels=["Wind_MWh_credit"]
columns_solar_features=["rad_t-1_dwd","rad_t_dwd","rad_t+1_dwd"]
columns_solar_labels=["Solar_MWh_credit"]

#风电光伏数据集
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
import pickle
with open('data/dataset/Dataset_stats.pkl', 'wb') as handle:
    pickle.dump(Dataset_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


#%% 导出数据集
IntegratedDataset.to_csv('data/dataset/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/SolarDataset.csv',index=False)

print("done!")


#%% --------------------------------------------数据集展示-------------------------------------------------------------

#展示功率-天气相关性
plt.figure(figsize=(9,5))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(data=IntegratedDataset, x="ws_100_t_dwd_1", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Generation [MWh]')

plt.figure(figsize=(9,5))
sns.scatterplot(data=IntegratedDataset, x="rad_t_dwd", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('Solar Radiation Downwards [w/m^2]')
plt.ylabel('Generation [MWh]')




#1.展示光伏容量变化
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
plt.figure(figsize=(12, 6))
plt.plot(energy_data["dtm"],energy_data["Solar_capacity_mwp"],label="Solar_capacity_mwp")
plt.xlabel("datetime",fontsize=25,family='Times New Roman')
plt.ylabel("Solar_capacity_mwp",fontsize=25,family='Times New Roman')
plt.xticks(fontsize=15,family='Times New Roman')
plt.yticks(fontsize=15,family='Times New Roman')
#设置图例字体
plt.legend(prop={'family':'Times New Roman','size':25})
plt.savefig("figs/Solar_capacity_mwp.png",dpi=660,bbox_inches='tight')
plt.show()


#2.展示每6个月年的光伏功率的峰值变化情况
energy_data_tmp=energy_data[["dtm","Wind_MWh_credit","Solar_MWh_credit","Solar_capacity_mwp"]]
energy_data_tmp['dtm'] = pd.to_datetime(energy_data['dtm'])
energy_data_tmp.set_index('dtm', inplace=True)
max_power_generation = energy_data_tmp.resample('6M').max()[['Solar_MWh_credit']]*2
plt.figure(figsize=(12, 6))
plt.plot(max_power_generation,marker='o',color='r',markersize=10,label="max_PV_power")
plt.xlabel("datetime",fontsize=25,family='Times New Roman')
plt.ylabel("MWh",fontsize=25,family='Times New Roman')
plt.xticks(fontsize=15,family='Times New Roman')
plt.yticks(fontsize=15,family='Times New Roman')
plt.legend(prop={'family':'Times New Roman','size':22})
plt.savefig("figs/max_power_generation.png",dpi=660,bbox_inches='tight')
plt.show()


#3.展示光伏每半小时功率变化
plt.figure(figsize=(12, 6))
plt.plot(energy_data["dtm"],energy_data["Solar_MWh_credit"]*2,label="Solar_MW")
plt.plot(dwd_solar_features["valid_datetime"],dwd_solar_features["SolarDownwardRadiation"],label="SolarDownwardRadiation")
plt.xlabel("datetime",fontsize=25,family='Times New Roman')
plt.ylabel("Solar_MWh_credit",fontsize=25,family='Times New Roman')
plt.legend(prop={'family':'Times New Roman','size':22})
plt.xticks(fontsize=15,family='Times New Roman')
plt.yticks(fontsize=15,family='Times New Roman')
plt.ylim(0,2000)
plt.savefig("figs/Solar_MWh_credit.png",dpi=660,bbox_inches='tight')
plt.show()


#4. 展示光伏功率/辐照度比例
plt.figure(figsize=(12, 6))
plt.plot(energy_data["dtm"],energy_data["Solar_MWh_credit"]*2/dwd_solar_features["SolarDownwardRadiation"],label="Solar_MW/SolarDownwardRadiation")
plt.xlabel("datetime",fontsize=25,family='Times New Roman')
plt.xticks(fontsize=15,family='Times New Roman')
plt.yticks(fontsize=15,family='Times New Roman')
plt.legend(prop={'family':'Times New Roman','size':22})
plt.savefig("figs/Solar_MW_SolarDownwardRadiation.png",dpi=660,bbox_inches='tight')
plt.show()



#"Solar_MWh_per_unit":energy_data.loc[i+1,"Solar_MWh_per_unit"],
'''
"ws_100_t-1_gfs_1":gfs_Hornsea1_features.loc[i,"WindSpeed:100"],
"ws_100_t-1_gfs_2":gfs_Hornsea1_features.loc[i,"WindSpeed:100^2"],
"ws_100_t-1_gfs_3":gfs_Hornsea1_features.loc[i,"WindSpeed:100^3"],
"wd_100_t-1_gfs_cos":np.cos(gfs_Hornsea1_features.loc[i,"WindDirection:100_rad"]),
"wd_100_t-1_gfs_sin":np.sin(gfs_Hornsea1_features.loc[i,"WindDirection:100_rad"]),
"temp_t-1_gfs":gfs_Hornsea1_features.loc[i,"Temperature"],

"ws_100_t_gfs_1":gfs_Hornsea1_features.loc[i+1,"WindSpeed:100"],
"ws_100_t_gfs_2":gfs_Hornsea1_features.loc[i+1,"WindSpeed:100^2"],
"ws_100_t_gfs_3":gfs_Hornsea1_features.loc[i+1,"WindSpeed:100^3"],
"wd_100_t_gfs_cos":np.cos(gfs_Hornsea1_features.loc[i+1,"WindDirection:100_rad"]),
"wd_100_t_gfs_sin":np.sin(gfs_Hornsea1_features.loc[i+1,"WindDirection:100_rad"]),
"temp_t_gfs":gfs_Hornsea1_features.loc[i+1,"Temperature"],

"ws_100_t+1_gfs_1":gfs_Hornsea1_features.loc[i+2,"WindSpeed:100"],
"ws_100_t+1_gfs_2":gfs_Hornsea1_features.loc[i+2,"WindSpeed:100^2"],
"ws_100_t+1_gfs_3":gfs_Hornsea1_features.loc[i+2,"WindSpeed:100^3"],
"wd_100_t+1_gfs_cos":np.cos(gfs_Hornsea1_features.loc[i+2,"WindDirection:100_rad"]),
"wd_100_t+1_gfs_sin":np.sin(gfs_Hornsea1_features.loc[i+2,"WindDirection:100_rad"]),
"temp_t+1_gfs":gfs_Hornsea1_features.loc[i+2,"Temperature"],

"rad_t-1_gfs":gfs_solar_features.loc[i,"SolarDownwardRadiation"],
"cloud_t-1_gfs":gfs_solar_features.loc[i,"CloudCover"],

"rad_t_gfs":gfs_solar_features.loc[i+1,"SolarDownwardRadiation"],
"cloud_t_gfs":gfs_solar_features.loc[i+1,"CloudCover"],

"rad_t+1_gfs":gfs_solar_features.loc[i+2,"SolarDownwardRadiation"],
"cloud_t+1_gfs":gfs_solar_features.loc[i+2,"CloudCover"],
'''