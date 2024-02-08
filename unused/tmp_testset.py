

#%% 处理dwd 风电数据
dwd_Hornsea1_old = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_new = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
dwd_Hornsea1_latest = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20240108_20240129.nc")


#特征提取
dwd_Hornsea1_features_old=dwd_Hornsea1_old["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(dwd_Hornsea1_old["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old.merge(dwd_Hornsea1_old["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

dwd_Hornsea1_features_new=dwd_Hornsea1_new["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(dwd_Hornsea1_new["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new.merge(dwd_Hornsea1_new["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])


#标准化时间
dwd_Hornsea1_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
dwd_Hornsea1_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
dwd_Hornsea1_features_old["ref_datetime"] = dwd_Hornsea1_features_old["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features_new["ref_datetime"] = dwd_Hornsea1_features_new["ref_datetime"].dt.tz_localize("UTC")


#提取valid_datetime>=23 且 valid_datetime<=47 的数据
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new[(dwd_Hornsea1_features_new["valid_datetime"]>=23) & (dwd_Hornsea1_features_new["valid_datetime"]<47)].reset_index(drop=True)
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old[(dwd_Hornsea1_features_old["valid_datetime"]>=23) & (dwd_Hornsea1_features_old["valid_datetime"]<47)].reset_index(drop=True)

#提取ref_datetime 的天为00:00的数据
dwd_Hornsea1_features_new=dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)
dwd_Hornsea1_features_old=dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)


#合并新旧数据
dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] <= "2023-10-26 00:00:00"]
dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new],axis=0).reset_index(drop=True)


#分别插值
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")
dwd_Hornsea1_features=dwd_Hornsea1_features.set_index("valid_datetime").resample("30T").interpolate("linear").reset_index()
#dwd_Hornsea1_features = dwd_Hornsea1_features.drop(columns="ref_datetime",axis=1).reset_index()

#提取风向的sin cos
dwd_Hornsea1_features["WindDirection:100_sin"]=np.sin(np.radians(dwd_Hornsea1_features["WindDirection:100"]))
dwd_Hornsea1_features["WindDirection:100_cos"]=np.cos(np.radians(dwd_Hornsea1_features["WindDirection:100"]))


#提取风速二次方和三次方
dwd_Hornsea1_features["WindSpeed:100^2"]=dwd_Hornsea1_features["WindSpeed:100"]**2
dwd_Hornsea1_features["WindSpeed:100^3"]=dwd_Hornsea1_features["WindSpeed:100"]**3


#%% 处理dwd 光伏数据
dwd_solar_old=xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_new=xr.open_dataset("data/dwd_icon_eu_pes10_20231027_20240108.nc")

#特征提取
dwd_solar_features_old=dwd_solar_old["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_old=dwd_solar_features_old.merge(dwd_solar_old["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])

dwd_solar_features_new=dwd_solar_new["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features_new=dwd_solar_features_new.merge(dwd_solar_new["CloudCover"].mean(dim="point").to_dataframe().reset_index(),how="outer",on=["reference_time","valid_time"])


#标准化时间
dwd_solar_features_new.rename(columns={"reference_time":"ref_datetime"},inplace=True)
dwd_solar_features_new.rename(columns={"valid_time":"valid_datetime"},inplace=True)
dwd_solar_features_old["ref_datetime"] = dwd_solar_features_old["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features_new["ref_datetime"] = dwd_solar_features_new["ref_datetime"].dt.tz_localize("UTC")

#提取valid_datetime>=23 且 valid_datetime<=47 的数据
dwd_solar_features_new=dwd_solar_features_new[(dwd_solar_features_new["valid_datetime"]>=23) & (dwd_solar_features_new["valid_datetime"]<47)].reset_index(drop=True)
dwd_solar_features_old=dwd_solar_features_old[(dwd_solar_features_old["valid_datetime"]>=23) & (dwd_solar_features_old["valid_datetime"]<47)].reset_index(drop=True)

#提取ref_datetime 的天为00:00的数据
dwd_solar_features_new=dwd_solar_features_new[dwd_solar_features_new["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)
dwd_solar_features_old=dwd_solar_features_old[dwd_solar_features_old["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)

#合并新旧数据
dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] <= "2023-10-26 00:00:00"]
dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new],axis=0).reset_index(drop=True)


#分别插值
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")
dwd_solar_features=dwd_solar_features.set_index("valid_datetime").resample("30T").interpolate("linear").reset_index()
#dwd_solar_features = dwd_solar_features.drop(columns="ref_datetime",axis=1).reset_index()



#%% 能源数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data_latest = pd.read_csv("data/energy_data_latest.csv")

#取2023-10-28 00:00 - 2024-01-08 05:00 时间段的数据，与天气预报数据对齐
energy_data_latest = energy_data_latest[energy_data_latest["dtm"] >= "2023-10-28 00:00"]
energy_data_latest = energy_data_latest[energy_data_latest["dtm"] < "2024-01-08 05:30"]
energy_data_latest["dtm"]=pd.to_datetime(energy_data_latest["dtm"])

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
modelling_table=dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
modelling_table= modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")

IntegratedDataset=pd.DataFrame()
for i in tqdm(range(len(modelling_table)-2)):
    data={

    "valid_datetime":modelling_table.loc[i+1,"valid_datetime"],
    "ref_datetime":modelling_table.loc[i+1,"ref_datetime"],
    
    "ws_100_t-1_dwd_1":modelling_table.loc[i,"WindSpeed:100"],
    "ws_100_t-1_dwd_2":modelling_table.loc[i,"WindSpeed:100^2"],
    "ws_100_t-1_dwd_3":modelling_table.loc[i,"WindSpeed:100^3"],
    "wd_100_t-1_dwd_cos":modelling_table.loc[i,"WindDirection:100_cos"],
    "wd_100_t-1_dwd_sin":modelling_table.loc[i,"WindDirection:100_sin"],
    "temp_t-1_dwd":modelling_table.loc[i,"Temperature"],
    
    
    "ws_100_t_dwd_1":modelling_table.loc[i+1,"WindSpeed:100"],
    "ws_100_t_dwd_2":modelling_table.loc[i+1,"WindSpeed:100^2"],
    "ws_100_t_dwd_3":modelling_table.loc[i+1,"WindSpeed:100^3"],
    "wd_100_t_dwd_cos":modelling_table.loc[i+1,"WindDirection:100_cos"],
    "wd_100_t_dwd_sin":modelling_table.loc[i+1,"WindDirection:100_sin"],
    "temp_t_dwd":modelling_table.loc[i+1,"Temperature"],


    "ws_100_t+1_dwd_1":modelling_table.loc[i+2,"WindSpeed:100"],
    "ws_100_t+1_dwd_2":modelling_table.loc[i+2,"WindSpeed:100^2"],
    "ws_100_t+1_dwd_3":modelling_table.loc[i+2,"WindSpeed:100^3"],
    "wd_100_t+1_dwd_cos":modelling_table.loc[i+2,"WindDirection:100_cos"],
    "wd_100_t+1_dwd_sin":modelling_table.loc[i+2,"WindDirection:100_sin"],
    "temp_t+1_dwd":modelling_table.loc[i+2,"Temperature"],
    

    "rad_t-1_dwd":modelling_table.loc[i,"SolarDownwardRadiation"],
    "cloud_t-1_dwd":modelling_table.loc[i,"CloudCover"],

    "rad_t_dwd":modelling_table.loc[i+1,"SolarDownwardRadiation"],
    "cloud_t_dwd":modelling_table.loc[i+1,"CloudCover"],

    "rad_t+1_dwd":modelling_table.loc[i+2,"SolarDownwardRadiation"],
    "cloud_t+1_dwd":modelling_table.loc[i+2,"CloudCover"],

    "Wind_MWh_credit":modelling_table.loc[i+1,"Wind_MWh_credit"],
    "Solar_MWh_credit":modelling_table.loc[i+1,"Solar_MWh_credit"], 
    "Solar_capacity_mwp":modelling_table.loc[i+1,"Solar_capacity_mwp"],
    "total_generation_MWh":modelling_table.loc[i+1,"Wind_MWh_credit"]+energy_data.loc[i+1,"Solar_MWh_credit"],
    }
   
    IntegratedDataset=IntegratedDataset._append(data,ignore_index=True)
    
    
#取后15%作为测试集
IntegratedDataset=IntegratedDataset.iloc[int(len(IntegratedDataset)*0.85):,:]

#%% 数据清洗

#缺失值处理
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')

#删除Wind_MWh_credit中超过650的异常值
IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<650]

#%% 特征工程

#提取valid_datetime中的小时
IntegratedDataset["hours"]=pd.to_datetime(IntegratedDataset["valid_datetime"]).dt.hour

#提取valid_datetime-ref_datetime的小时差,以2小时为间隔
IntegratedDataset["hours_diff"]=pd.to_datetime(IntegratedDataset["valid_datetime"])-pd.to_datetime(IntegratedDataset["ref_datetime"])
IntegratedDataset["hours_diff"]=IntegratedDataset["hours_diff"].dt.total_seconds()/3600
IntegratedDataset["hours_diff"]=IntegratedDataset["hours_diff"].astype(int)
IntegratedDataset["hours_diff"]=(IntegratedDataset["hours_diff"]/2).astype(int)


#IntegratedDataset=pd.read_csv("data/dataset/test/IntegratedDataset.csv")
#%% 风电和光伏预测分别所需的列名

#从训练的IntegratedDataset去读中提取风电和光伏数据集所需的列名


columns_wind_features=["ws_100_t-1_dwd_1",
                       "ws_100_t_dwd_1",
                       "ws_100_t+1_dwd_1"]
columns_wind_labels=["Wind_MWh_credit"]
columns_solar_features=["rad_t-1_dwd","rad_t_dwd","rad_t+1_dwd","hours"]
columns_solar_labels=["Solar_MWh_credit"]


#风电光伏数据集
WindDataset=IntegratedDataset[columns_wind_features+columns_wind_labels]
SolarDataset=IntegratedDataset[columns_solar_features+columns_solar_labels]

#%% 保存数据集
IntegratedDataset.to_csv('data/dataset/test/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/test/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/test/SolarDataset.csv',index=False)

print("done!")




