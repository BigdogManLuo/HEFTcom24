import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import comp_utils
from comp_utils import pinball
import forecaster

#登录
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

#获取过去提交的预测数据
submissions=rebase_api_client.get_submissions()

solution=submissions["items"][29]["solution"]
print(f"market day: {solution['market_day']}")

submission=solution['submission']

Total_Generation_Forecast={f"q{quantile}":[] for quantile in range(10,100,10)}

for quantile in range(10,100,10):
    for i in range(48):
        Total_Generation_Forecast[f"q{quantile}"].append(submission[i]["probabilistic_forecast"][f'{quantile}'])


#读取真实发电数据
wind_generation=rebase_api_client.get_variable(day="2024-01-23",variable="wind_total_production")
wind_generation["Wind_MWh_Credit"]=0.5*wind_generation["generation_mw"]-wind_generation["boa"]

solar_generation=rebase_api_client.get_variable(day="2024-01-25",variable="solar_total_production")
solar_generation["Solar_MWh_Credit"]=0.5*solar_generation["generation_mw"]

Total_Generation_True=wind_generation["Wind_MWh_Credit"]+solar_generation["Solar_MWh_Credit"]

#TMP!!!
'''
Total_Generation_Forecast["q10"]=[x-300 for x in Total_Generation_Forecast["q10"]]
Total_Generation_Forecast["q20"]=[x-200 for x in Total_Generation_Forecast["q20"]]
Total_Generation_Forecast["q30"]=[x-100 for x in Total_Generation_Forecast["q30"]]
'''
#计算平均pinball损失
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=Total_Generation_True[0:48],y_hat=Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"pinball_score: {pinball_score}")

#绘制图像
plt.figure(figsize=(8,6))
for quantile in range(10,100,10):
    plt.plot(Total_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
    
plt.plot(Total_Generation_True[0:48],label="True")
plt.legend()
plt.title(f"{solution['market_day']}")
plt.show()

#%% 调用当前模型测试

#-------------------------------获取当天天气特征--------------------------------
source="dwd"
IntegratedDataset=pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")
#获取ref_datetime为2024-01-xx 00:00:00的数据
IntegratedDataset["ref_datetime"] = pd.to_datetime(IntegratedDataset["ref_datetime"])
IntegratedDataset["valid_datetime"] = pd.to_datetime(IntegratedDataset["valid_datetime"])
IntegratedDataset=IntegratedDataset[IntegratedDataset["ref_datetime"].dt.strftime("%Y-%m-%d %H:%M")=="2024-01-21 00:00"].reset_index(drop=True)

#获取valid_datetime与ref_datetime的差值为23小时~47小时的数据
IntegratedDataset=IntegratedDataset[(IntegratedDataset["valid_datetime"]-IntegratedDataset["ref_datetime"])>=np.timedelta64(23,"h")].reset_index(drop=True)
#检查一下是否跟api获取的数据一致

#构造风电光伏特征
columns_wind_features=["ws_100_t_dwd_1","ws_100_t_dwd_max","ws_100_t_dwd_min","ws_100_t_dwd_q75","ws_100_t_dwd_q25",
                            "ws_100_t+1_dwd_1","ws_100_t+1_dwd_max","ws_100_t+1_dwd_min","ws_100_t+1_dwd_q75","ws_100_t+1_dwd_q25"]
columns_solar_features=["rad_t-1_dwd","rad_t-1_dwd_max","rad_t-1_dwd_min","rad_t-1_dwd_q75","rad_t-1_dwd_q25",
                        "rad_t_dwd","rad_t_dwd_max","rad_t_dwd_min","rad_t_dwd_q75","rad_t_dwd_q25","hours"]
Features_wind=IntegratedDataset[columns_wind_features]
Features_solar=IntegratedDataset[columns_solar_features]

#归一化
#z-score标准化(仅对特征)
with open(f"data/dataset/{source}/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

features_wind=(Features_wind-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]
features_solar=Features_solar.copy()
features_solar.iloc[:,0:-1]=(Features_solar.iloc[:,0:-1]-Dataset_stats["Mean"]["features"]["solar"])/Dataset_stats["Std"]["features"]["solar"]

features_wind=np.array(features_wind)
features_solar=np.array(features_solar)

#提取时间序列特征
hours=IntegratedDataset["hours"]

#----------------------------------调用模型预测----------------------------------
Total_Generation_Forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=forecaster.forecastByLGBM(features_wind,features_solar,Dataset_stats,hours,full=False)

#计算平均pinball损失
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=Total_Generation_True,y_hat=Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"pinball_score: {pinball_score}")

#------------------------------------可视化发电曲线--------------------------
plt.figure(figsize=(8,6))

#总预测结果
plt.subplot(3,1,1)
plt.title("Total",fontsize=18)
plt.plot(Total_Generation_True[0:48],label="true")
plt.plot(Total_Generation_Forecast["q50"],label="forecast")
plt.legend()
plt.grid()

    
#光伏预测结果
plt.subplot(3,1,2)
plt.title("Solar",fontsize=18)
plt.plot(solar_generation["Solar_MWh_Credit"],label="true")
plt.plot(Solar_Generation_Forecast["q50"],label="forecast")
plt.legend()
plt.grid()

#风电预测结果
plt.subplot(3,1,3)
plt.title("Wind",fontsize=18)
plt.plot(wind_generation["Wind_MWh_Credit"],label="true")
plt.plot(Wind_Generation_Forecast["q50"],label="forecast")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

#-------------------------可视化天气特征与发电相关关系------------------------------
plt.figure(figsize=(8,10))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.scatter(Features_wind.iloc[:,i],Wind_Generation_Forecast["q50"])
    plt.ylabel("q50")
    plt.xlabel(Features_wind.columns[i])


#光伏预测结果
#横坐标为dwd_solar_features的每一列，共3列
#纵坐标为submission_data的q50列
plt.figure(figsize=(8,10))
for i in range(3):
    plt.subplot(3,1,i+1)
    plt.scatter(Features_solar.iloc[:,i],Solar_Generation_Forecast["q50"])
    plt.ylabel("q50")
    plt.xlabel(Features_solar.columns[i])
