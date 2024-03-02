import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import comp_utils
from comp_utils import pinball
from forecaster import forecast

#登录
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

day="2024-02-22"

#=======================获取过去提交的预测数据=====================================
submissions=rebase_api_client.get_submissions(market_day=day)
solution=submissions["items"][0]["solution"]
print(f"market day: {solution['market_day']}")
submission=solution['submission']

Total_Generation_Forecast={f"q{quantile}":[] for quantile in range(10,100,10)}

for quantile in range(10,100,10):
    for i in range(48):
        Total_Generation_Forecast[f"q{quantile}"].append(submission[i]["probabilistic_forecast"][f'{quantile}'])


#=======================获取当天真实发电数据=====================================
wind_generation=rebase_api_client.get_variable(day=day,variable="wind_total_production")
wind_generation["Wind_MWh_Credit"]=0.5*wind_generation["generation_mw"]-wind_generation["boa"]


solar_generation=rebase_api_client.get_variable(day=day,variable="solar_total_production")
solar_generation["Solar_MWh_Credit"]=0.5*solar_generation["generation_mw"]

Total_Generation_True=wind_generation["Wind_MWh_Credit"]+solar_generation["Solar_MWh_Credit"]
Total_Generation_True=Total_Generation_True[0:48]

#计算平均pinball损失
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=Total_Generation_True,y_hat=Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"pinball_score: {pinball_score}")

#绘制图像
plt.figure(figsize=(8,6))
for quantile in range(10,100,10):
    plt.plot(Total_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
    
plt.plot(Total_Generation_True,label="True")
plt.legend()
plt.title(f"{solution['market_day']}")
plt.show()

#%% 调用当前模型测试

#-------------------------------获取当天天气特征--------------------------------
source="dwd"

IntegratedFeatures=pd.read_csv(f"logs/dfs/{day}_IntegratedFeatures.csv")
columns_wind=pd.read_csv("data/dataset/dwd/WindDataset.csv").columns.tolist()[0:-1]
columns_solar=pd.read_csv("data/dataset/dwd/SolarDataset.csv").columns.tolist()[0:-1]
Features_wind=IntegratedFeatures[columns_wind]
Features_solar=IntegratedFeatures[columns_solar]

#归一化
with open(f"data/dataset/{source}/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

features_wind=(Features_wind-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]
features_solar=Features_solar.copy()
features_solar.iloc[:,0:-1]=(Features_solar.iloc[:,0:-1]-Dataset_stats["Mean"]["features"]["solar"])/Dataset_stats["Std"]["features"]["solar"]
features_wind=np.array(features_wind)
features_solar=np.array(features_solar)

#----------------------------------调用模型预测----------------------------------
params={
    "wind_features":features_wind,
    "solar_features":features_solar,
    "Dataset_stats":Dataset_stats,
    "hours":features_solar[:,-1],
    "model_name":"LGBM",
    "full":True,
    "WLimit":True,
    "maxPower":220
}
Total_Generation_Forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=forecast(**params)



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
