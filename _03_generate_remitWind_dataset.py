import pandas as pd
import numpy as np
import glob
import comp_utils
from tqdm import tqdm
#获取天气特征
logs=glob.glob("logs/dfs/*.csv")
IntegratedFeatures=pd.concat((pd.read_csv(log) for log in logs))
IntegratedFeatures["valid_datetime"]=pd.to_datetime(IntegratedFeatures["valid_datetime"]).dt.tz_convert('UTC')
IntegratedFeatures["ref_datetime"]=pd.to_datetime(IntegratedFeatures["ref_datetime"]).dt.tz_convert('UTC')
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

energy_data_latest=pd.DataFrame()
for day in tqdm(pd.date_range("2024-02-05","2024-02-21")):
    day=day.tz_localize("UTC")
    try:
        #获取当天的风电数据
        wind_tmp=rebase_api_client.get_variable(day=day.strftime("%Y-%m-%d"),variable="wind_total_production")
        wind_tmp["timestamp_utc"]=pd.to_datetime(wind_tmp["timestamp_utc"])

    except:
        #创建一个含nan的dataframe
        columns=["timestamp_utc","generation_mw","boa"]
        wind_tmp=pd.DataFrame(columns=columns)
        print(f"get wind data failed: {day}")

    try:
        #获取当天的光伏数据
        solar_tmp=rebase_api_client.get_variable(day=day.strftime("%Y-%m-%d"),variable="solar_total_production")
        solar_tmp["timestamp_utc"]=pd.to_datetime(solar_tmp["timestamp_utc"])

    except:
        #创建一个含nan的dataframe
        columns=["timestamp_utc","generation_mw","installed_capacity_mwp","capacity_mwp"]
        solar_tmp=pd.DataFrame(columns=columns)
        print(f"get solar data failed: {day}")


    # 将当天的风电数据和光伏数据加入到energy_data_latest
    energy_data_today=pd.DataFrame({"dtm": pd.to_datetime(wind_tmp["timestamp_utc"]),
        "boa": wind_tmp["boa"],
        "Wind_MW": wind_tmp["generation_mw"],
        "Solar_MW": solar_tmp["generation_mw"],
        "installed_capacity_mwp": solar_tmp["installed_capacity_mwp"],
        "capacity_mwp": solar_tmp["capacity_mwp"]
        })
    
    energy_data_latest = pd.concat([energy_data_latest, energy_data_today], ignore_index=True)

energy_data_latest["dtm"] = pd.to_datetime(energy_data_latest["dtm"])
energy_data_latest['dtm'] = energy_data_latest['dtm'].dt.tz_convert('UTC')
energy_data_latest.rename(columns={"capacity_mwp":"Solar_capacity_mwp"},inplace=True)

energy_data_latest["Wind_MWh_credit"] = 0.5*energy_data_latest["Wind_MW"] - energy_data_latest["boa"]
energy_data_latest["Solar_MWh_credit"] = 0.5*energy_data_latest["Solar_MW"]

import matplotlib.pyplot as plt
energy_data_latest=energy_data_latest.dropna(axis=0,how='any')
plt.plot(energy_data_latest["Wind_MWh_credit"])



#合并天气数据和发电数据
IntegratedDataset = IntegratedFeatures.merge(energy_data_latest,how="inner",left_on="valid_datetime",right_on="dtm")

#缺失值处理
IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')

#获取风电限电数据集所需要的列
columns_wind=pd.read_csv("data/dataset/dwd/WindDataset.csv").columns.tolist()
WindRemitDataset=IntegratedDataset[columns_wind]

#保存
WindRemitDataset.to_csv("data/dataset/dwd/WindRemitDataset.csv",index=False)