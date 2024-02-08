import numpy as np
import pandas as pd
import lightgbm as lgb
from comp_utils import pinball
from matplotlib import pyplot as plt

#读取数据集
WindDataset=pd.read_csv("data/dataset/WindDataset.csv")
SolarDataset=pd.read_csv("data/dataset/SolarDataset.csv")

#提取特征和标签
features_wind=np.array(WindDataset.iloc[:,:-1])
labels_wind=np.array(WindDataset.iloc[:,-1])
features_solar=np.array(SolarDataset.iloc[:,:-1])
labels_solar=np.array(SolarDataset.iloc[:,-1])

#加载原风电、光伏数据集的均值和方差
Mean_features_wind=np.load("data/dataset/Mean_features_wind.npy")
Std_features_wind=np.load("data/dataset/Std_features_wind.npy")
Mean_features_solar=np.load("data/dataset/Mean_features_solar.npy")
Std_features_solar=np.load("data/dataset/Std_features_solar.npy")
Mean_labels_wind=np.load("data/dataset/Mean_labels_wind.npy")
Std_labels_wind=np.load("data/dataset/Std_labels_wind.npy")
Mean_labels_solar=np.load("data/dataset/Mean_labels_solar.npy")
Std_labels_solar=np.load("data/dataset/Std_labels_solar.npy")

#z-score标准化(仅对特征)
features_wind=(features_wind-Mean_features_wind)/Std_features_wind
features_solar=(features_solar-Mean_features_solar)/Std_features_solar

#取后20%作为测试集
features_wind=features_wind[int(0.8*len(features_wind)):]
labels_wind=labels_wind[int(0.8*len(labels_wind)):]
features_solar=features_solar[int(0.8*len(features_solar)):]
labels_solar=labels_solar[int(0.8*len(labels_solar)):]

#加载模型
LGBM_models_wind={}
LGBM_models_solar={}
for quantile in range(10,100,10):
    bst_wind = lgb.Booster(model_file=f"models/LGBM/wind_q{quantile}.txt")
    LGBM_models_wind[f"q{quantile}"]=bst_wind
    bst_solar = lgb.Booster(model_file=f"models/LGBM/solar_q{quantile}.txt")
    LGBM_models_solar[f"q{quantile}"]=bst_solar

#测试
Wind_Generation_forecast=dict()
Solar_Generation_forecast=dict()
Total_Generation_forecast=dict()
pinball_losses=dict()

for quantile in range(10,100,10):

    #前向
    output_wind= LGBM_models_wind[f"q{quantile}"].predict(features_wind)
    output_solar= LGBM_models_solar[f"q{quantile}"].predict(features_solar)
    
    #逆归一化
    output_wind=output_wind*Std_labels_wind+Mean_labels_wind
    output_solar=output_solar*Std_labels_solar+Mean_labels_solar

    #手动修正
    output_wind[output_wind<0]=0
    output_solar[output_solar<1e-2]=0

    #计算损失
    total_output=output_wind+output_solar #总发电量预测
    labels=labels_wind+labels_solar #总发电量真实值

    pinball_losses[f"q{quantile}"] = pinball(y_hat=total_output,
                                                y=labels,
                                                alpha=quantile/100)
    
    #记录预测结果
    Wind_Generation_forecast[f"q{quantile}"]=output_wind
    Solar_Generation_forecast[f"q{quantile}"]=output_solar
    Total_Generation_forecast[f"q{quantile}"]=total_output


#绘图
plt.figure(figsize=(15,10))
x_range=np.arange(1000)

#总预测结果
plt.subplot(3,1,1)
plt.plot(x_range,labels_wind[x_range]+labels_solar[x_range],label="true")
plt.plot(x_range,Total_Generation_forecast["q50"][0:len(x_range)],label="q50")

#光伏预测结果
plt.subplot(3,1,2)
plt.plot(x_range,labels_solar[x_range],label="true")
plt.plot(x_range,Solar_Generation_forecast["q50"][0:len(x_range)],label="q50")

#风电预测结果
plt.subplot(3,1,3)
plt.plot(x_range,labels_wind[x_range],label="true")
plt.plot(x_range,Wind_Generation_forecast["q50"][0:len(x_range)],label="q50")

plt.show()

#打印平均损失
print("Score:",np.mean(list(pinball_losses.values())))
