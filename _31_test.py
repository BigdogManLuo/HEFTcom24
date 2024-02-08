import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from comp_utils import pinball
from forecaster import forecast
import pickle

'''-------------------------------测试摘要------------------------------'''
recordings="""
天气源:单天气源dwd
利用预报时段：前48h预报数据
风电特征: t~t+1时段 100m风速(mean,max,min,q75,q25)
光伏特征: t-1~t时段 辐照度(mean,max,min,q75,q25)+季节
发电数据：清除风电异常数据
模型: LightGBM
手动工程：光伏夜晚补0，9个模型超参数调优（未完全调整）
"""

source="dwd"

#加载数据集
SolarDataset=pd.read_csv("data/dataset/test/SolarDataset.csv")
WindDataset=pd.read_csv("data/dataset/test/WindDataset.csv")

#提取特征和标签
features_wind=WindDataset.iloc[:,:-1]
labels_wind=WindDataset.iloc[:,-1]
features_solar=SolarDataset.iloc[:,:-1]
labels_solar=SolarDataset.iloc[:,-1]

#z-score标准化(仅对特征)
with open(f"data/dataset/{source}/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

features_wind=(features_wind-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]
features_solar.iloc[:,0:-1]=(features_solar.iloc[:,0:-1]-Dataset_stats["Mean"]["features"]["solar"])/Dataset_stats["Std"]["features"]["solar"]

features_wind=np.array(features_wind)
features_solar=np.array(features_solar)
labels_wind=np.array(labels_wind)
labels_solar=np.array(labels_solar)

#提取时间序列特征
IntegratedDataset=pd.read_csv('data/dataset/test/IntegratedDataset.csv')
hours=IntegratedDataset["hours"]


#%% 预测
params={
    "wind_features":features_wind,
    "solar_features":features_solar,
    "Dataset_stats":Dataset_stats,
    "hours":hours,
    "full":False,
    "model_name":"LGBM",
    "WLimit":False
}
Total_Generation_Forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=forecast(**params)

print(recordings)
#计算分位数损失
Pinball_Losses_Total={}
Pinball_Losses_Wind={}
Pinball_Losses_Solar={}
for quantile in range(10,100,10):
    Pinball_Losses_Total[f"q{quantile}"]=pinball(labels_wind+labels_solar,Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100)

#计算光伏分位数损失
for quantile in range(10,100,10):
    Pinball_Losses_Solar[f"q{quantile}"]=pinball(labels_solar,Solar_Generation_Forecast[f"q{quantile}"],alpha=quantile/100)

#计算风电分位数损失
for quantile in range(10,100,10):
    Pinball_Losses_Wind[f"q{quantile}"]=pinball(labels_wind,Wind_Generation_Forecast[f"q{quantile}"],alpha=quantile/100)
    
print(f"Total Score:{sum(Pinball_Losses_Total.values())/len(Pinball_Losses_Total.values())}")
print(f"Wind Score:{sum(Pinball_Losses_Wind.values())/len(Pinball_Losses_Wind.values())}")
print(f"Solar Score:{sum(Pinball_Losses_Solar.values())/len(Pinball_Losses_Solar.values())}")


#%% 绘图
plt.figure(figsize=(8,6))
x_range=np.arange(0,1000)

#总预测结果
plt.subplot(3,1,1)
plt.title("Total")
plt.plot(labels_wind[x_range]+labels_solar[x_range],label="true")
plt.plot(Total_Generation_Forecast["q50"][x_range[0]:x_range[-1]],label="q50")
plt.legend()
plt.grid()

    
#光伏预测结果
plt.subplot(3,1,2)
plt.title("Solar")
plt.plot(labels_solar[x_range],label="true")
plt.plot(Solar_Generation_Forecast["q50"][x_range[0]:x_range[-1]],label="q50")
plt.legend()
plt.grid()

#风电预测结果
plt.subplot(3,1,3)
plt.title("Wind")
plt.plot(labels_wind[x_range],label="true")
plt.plot(Wind_Generation_Forecast["q50"][x_range[0]:x_range[-1]],label="q50")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


#%% 记录每次提交的代码改动记录
flag=input("是否记录(Y/n):")
if flag=="Y":
    text_file = open(f"logs/tests/sub_{pd.Timestamp('today').strftime('%Y%m%d-%H%M%S')}.txt", "w")
    text_file.write(f"Total Score:{sum(Pinball_Losses_Total.values())/len(Pinball_Losses_Total.values())}")
    text_file.write("\n")
    text_file.write(f"Wind Score:{sum(Pinball_Losses_Wind.values())/len(Pinball_Losses_Wind.values())}")
    text_file.write("\n")
    text_file.write(f"Solar Score:{sum(Pinball_Losses_Solar.values())/len(Pinball_Losses_Solar.values())}")
    text_file.write("\n")
    text_file.write(recordings)
    text_file.close()
elif flag=="n":
    pass
else:
    raise ValueError("输入错误")








