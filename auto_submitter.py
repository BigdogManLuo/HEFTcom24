import comp_utils
import pandas as pd
import numpy as np
from comp_utils import send_mes
import matplotlib.pyplot as plt
from forecaster import forecast,forecastByBenchmark,forecastByStacking,adjust_forecast
import pickle
'''-------------------------------提交摘要------------------------------'''
recordings="""
天气源:单天气源dwd
利用预报时段：前48h预报数据
风电特征: t~t+1时段 100m风速(mean,max,min,q75,q25)
光伏特征: t-1~t时段 辐照度(mean,max,min,q75,q25)+hour
发电数据：清除风电异常数据
模型: LightGBM
手动工程：光伏夜晚补0，9个模型超参数调优（未完全调整）
风电限电：限电模型学习
"""


#读取队伍API key
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

#获取最新天气数据
latest_dwd_Hornsea1 = comp_utils.weather_df_to_xr(rebase_api_client.get_hornsea_dwd())
latest_dwd_solar = comp_utils.weather_df_to_xr(rebase_api_client.get_pes10_nwp("DWD_ICON-EU"))
#latest_gfs_Hornsea1 = comp_utils.weather_df_to_xr(rebase_api_client.get_hornsea_gfs())
#latest_gfs_solar = comp_utils.weather_df_to_xr(rebase_api_client.get_pes10_nwp("NCEP_GFS"))

#===========================================特征提取=================================================
#平均风速
latest_dwd_Hornsea1_features=latest_dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
#最大风速
latest_dwd_Hornsea1_features=latest_dwd_Hornsea1_features.merge(
    latest_dwd_Hornsea1["WindSpeed:100"].max(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"maxWindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"])
#最小风速
latest_dwd_Hornsea1_features=latest_dwd_Hornsea1_features.merge(
    latest_dwd_Hornsea1["WindSpeed:100"].min(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"minWindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"]) 
#75%分位数风速
latest_dwd_Hornsea1_features=latest_dwd_Hornsea1_features.merge(
    latest_dwd_Hornsea1["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.75).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q75WindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"])
#25%分位数风速
latest_dwd_Hornsea1_features=latest_dwd_Hornsea1_features.merge(
    latest_dwd_Hornsea1["WindSpeed:100"].quantile(dim=["latitude","longitude"],q=0.25).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"q25WindSpeed:100"}),
    how="left",on=["ref_datetime","valid_datetime"])

#平均辐照度
latest_dwd_solar_features=latest_dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
#最大辐照度
latest_dwd_solar_features=latest_dwd_solar_features.merge(
    latest_dwd_solar["SolarDownwardRadiation"].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])
#最小辐照度
latest_dwd_solar_features=latest_dwd_solar_features.merge(
    latest_dwd_solar["SolarDownwardRadiation"].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])
#75%分位数辐照度
latest_dwd_solar_features=latest_dwd_solar_features.merge(
    latest_dwd_solar["SolarDownwardRadiation"].quantile(dim="point",q=0.75).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q75SolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])
#25%分位数辐照度
latest_dwd_solar_features=latest_dwd_solar_features.merge(
    latest_dwd_solar["SolarDownwardRadiation"].quantile(dim="point",q=0.25).to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"q25SolarDownwardRadiation"}),
    how="left",on=["ref_datetime","valid_datetime"])


#%%===========================================数据预处理=================================================

#合并风光特征
latest_forecast_table = latest_dwd_Hornsea1_features.merge(latest_dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])

#插值为半小时分辨率
latest_forecast_table = latest_forecast_table.set_index("valid_datetime").resample("30T").interpolate("linear",limit=5).reset_index()

#定位提交时间
submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})

#在submission_data的datetime列的首尾分别添加上前半小时的时间和后半小时的时间
features_datetime=pd.concat([
    pd.DataFrame({"datetime":submission_data.iloc[0:1,0]-pd.Timedelta("30min")}),
    submission_data,
    pd.DataFrame({"datetime":submission_data.iloc[-1:,0]+pd.Timedelta("30min")})]
    ,axis=0).reset_index()

#按照features_datetime提取latest_forecast_table
latest_forecast_table=latest_forecast_table[latest_forecast_table["valid_datetime"].isin(features_datetime["datetime"])].reset_index()

#创建特定格式数据集
IntegratedFeatures=pd.DataFrame()
columns_wind_features=pd.read_csv("data/dataset/dwd/WindDataset.csv").columns.tolist()[:-1]
columns_solar_features=pd.read_csv("data/dataset/dwd/SolarDataset.csv").columns.tolist()[:-1]

for i in range(len(latest_forecast_table)-2):

    feature={
        
        "ws_100_t-1_dwd_1":latest_forecast_table.iloc[i]["WindSpeed:100"],
        "ws_100_t-1_dwd_max":latest_forecast_table.iloc[i]["maxWindSpeed:100"],
        "ws_100_t-1_dwd_min":latest_forecast_table.iloc[i]["minWindSpeed:100"],
        "ws_100_t-1_dwd_q75":latest_forecast_table.iloc[i]["q75WindSpeed:100"],
        "ws_100_t-1_dwd_q25":latest_forecast_table.iloc[i]["q25WindSpeed:100"],
        
        "ws_100_t_dwd_1":latest_forecast_table.iloc[i+1]["WindSpeed:100"],
        "ws_100_t_dwd_max":latest_forecast_table.iloc[i+1]["maxWindSpeed:100"],
        "ws_100_t_dwd_min":latest_forecast_table.iloc[i+1]["minWindSpeed:100"],
        "ws_100_t_dwd_q75":latest_forecast_table.iloc[i+1]["q75WindSpeed:100"],
        "ws_100_t_dwd_q25":latest_forecast_table.iloc[i+1]["q25WindSpeed:100"],
        
        "ws_100_t+1_dwd_1":latest_forecast_table.iloc[i+2]["WindSpeed:100"],
        "ws_100_t+1_dwd_max":latest_forecast_table.iloc[i+2]["maxWindSpeed:100"],
        "ws_100_t+1_dwd_min":latest_forecast_table.iloc[i+2]["minWindSpeed:100"],
        "ws_100_t+1_dwd_q75":latest_forecast_table.iloc[i+2]["q75WindSpeed:100"],
        "ws_100_t+1_dwd_q25":latest_forecast_table.iloc[i+2]["q25WindSpeed:100"],
        
        "rad_t-1_dwd":latest_forecast_table.iloc[i]["SolarDownwardRadiation"],
        "rad_t-1_dwd_max":latest_forecast_table.iloc[i]["maxSolarDownwardRadiation"],
        "rad_t-1_dwd_min":latest_forecast_table.iloc[i]["minSolarDownwardRadiation"],
        "rad_t-1_dwd_q75":latest_forecast_table.iloc[i]["q75SolarDownwardRadiation"],
        "rad_t-1_dwd_q25":latest_forecast_table.iloc[i]["q25SolarDownwardRadiation"],            
        
        "rad_t_dwd":latest_forecast_table.iloc[i+1]["SolarDownwardRadiation"],
        "rad_t_dwd_max":latest_forecast_table.iloc[i+1]["maxSolarDownwardRadiation"],
        "rad_t_dwd_min":latest_forecast_table.iloc[i+1]["minSolarDownwardRadiation"],
        "rad_t_dwd_q75":latest_forecast_table.iloc[i+1]["q75SolarDownwardRadiation"],
        "rad_t_dwd_q25":latest_forecast_table.iloc[i+1]["q25SolarDownwardRadiation"],
        
        "rad_t+1_dwd":latest_forecast_table.iloc[i+2]["SolarDownwardRadiation"],
        "rad_t+1_dwd_max":latest_forecast_table.iloc[i+2]["maxSolarDownwardRadiation"],
        "rad_t+1_dwd_min":latest_forecast_table.iloc[i+2]["minSolarDownwardRadiation"],
        "rad_t+1_dwd_q75":latest_forecast_table.iloc[i+2]["q75SolarDownwardRadiation"],
        "rad_t+1_dwd_q25":latest_forecast_table.iloc[i+2]["q25SolarDownwardRadiation"],
        
        "ref_datetime":latest_forecast_table.iloc[i+1]["ref_datetime"],
        "valid_datetime":latest_forecast_table.iloc[i+1]["valid_datetime"],
    }

    IntegratedFeatures=IntegratedFeatures._append(feature,ignore_index=True)

IntegratedFeatures["hours"]=pd.to_datetime(IntegratedFeatures["valid_datetime"]).dt.hour
hours=IntegratedFeatures["hours"]

columns_wind_features=pd.read_csv("data/dataset/dwd/WindDataset.csv").columns.tolist()[:-1]
columns_solar_features=pd.read_csv("data/dataset/dwd/SolarDataset.csv").columns.tolist()[:-1]
dwd_wind_features=IntegratedFeatures[columns_wind_features]
dwd_solar_features=IntegratedFeatures[columns_solar_features]

wind_features=dwd_wind_features.copy()
solar_features=dwd_solar_features.copy()
#z-score标准化
with open("data/dataset/dwd/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

wind_features=(dwd_wind_features-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]
solar_features.iloc[:,0:-1]=(dwd_solar_features.iloc[:,0:-1]-Dataset_stats["Mean"]["features"]["solar"])/Dataset_stats["Std"]["features"]["solar"]

#校验列名
print(columns_wind_features==wind_features.columns)
print(columns_solar_features==solar_features.columns)

#转换为numpy数组
wind_features=np.array(wind_features)
solar_features=np.array(solar_features)


#%% Pre-Forecast
wind_forecast_table=dwd_wind_features["ws_100_t_dwd_1"].to_frame()
wind_forecast_table.rename(columns={"ws_100_t_dwd_1":"WindSpeed"},inplace=True)
solar_forecat_table=dwd_solar_features["rad_t_dwd"].to_frame()
solar_forecat_table.rename(columns={"rad_t_dwd":"SolarDownwardRadiation"},inplace=True)


pre_Total_Generation_Forecast,pre_Wind_Generation_Forecast,pre_Solar_Generation_Forecast=forecastByBenchmark(wind_forecast_table,solar_forecat_table)


#%% 预测
params={
    "wind_features":wind_features,
    "solar_features":solar_features,
    "Dataset_stats":Dataset_stats,
    "hours":hours,
    "model_name":"LGBM",
    "full":True,
    "WLimit":True,
    "maxPower":410
}
_,Wind_Generation_Forecast,_=forecast(**params)

#用集成学习模型预测光伏
params={
    "wind_features":wind_features,
    "solar_features":solar_features,
    "Dataset_stats":Dataset_stats,
    "hours":hours,
    "full":False,
}
_,_,Solar_Generation_Forecast=forecastByStacking(**params)

Total_Generation_Forecast={}
for quantile in range(10,100,10):
    Total_Generation_Forecast[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]+Solar_Generation_Forecast[f"q{quantile}"]

#分位数重新排序，确保大的分位数结果更大
Total_Generation_Forecast=adjust_forecast(Total_Generation_Forecast)

#合成提交数据
for quantile in range(10,100,10):
    submission_data[f"q{quantile}"]=Total_Generation_Forecast[f"q{quantile}"]


#%% 投标
#根据电价预测结果
submission_data["market_bid"]=submission_data["q50"]

#%% 日志记录
#设置tomorrow为明天的日期
tomorrow=pd.Timestamp.now().date()+pd.Timedelta("1D")
IntegratedFeatures.to_csv(f"logs/dfs/{tomorrow}_IntegratedFeatures.csv",index=False)


#%%=====================================预测结果=====================================
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(Wind_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',frameon=False)
plt.ylabel("Wind Generation")
plt.xlabel("Time")
plt.grid()
plt.subplot(3,1,2)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(Solar_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
plt.ylabel("Solar Generation")
plt.xlabel("Time")
plt.grid()
plt.subplot(3,1,3)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(submission_data[f"q{quantile}"],label=f"q{quantile}")
plt.ylabel("TotalGeneration")
plt.xlabel("Time")
plt.grid()

plt.savefig(f"logs/figs/{tomorrow}_forecast.png",dpi=500)

#%% =====================================与benchmark的预测结果对比=================================
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.plot(pre_Wind_Generation_Forecast["q50"],label="benchmark")
plt.plot(Wind_Generation_Forecast["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Wind Generation")
plt.grid()

plt.subplot(3,1,2)
plt.plot(pre_Solar_Generation_Forecast["q50"],label="benchmark")
plt.plot(Solar_Generation_Forecast["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Solar Generation")
plt.grid()

plt.subplot(3,1,3)
plt.plot(pre_Total_Generation_Forecast["q50"],label="benchmark")
plt.plot(submission_data["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Total Generation")
plt.grid()
plt.tight_layout()
plt.savefig(f"logs/figs/{tomorrow}_benchmark_comp.png",dpi=500)

#%% 提交

submission_data = comp_utils.prep_submission_in_json_format(submission_data)

resp=rebase_api_client.submit(submission_data,recordings) #最终提交
submissions=rebase_api_client.get_submissions(market_day=tomorrow)
send_mes(recordings,resp,tomorrow)
if resp.status_code==200:
    print("提交成功")
else:
    raise ValueError("提交失败")