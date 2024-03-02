import pandas as pd
import pickle
import matplotlib.pyplot as plt
from comp_utils import pinball
from lightgbm import LGBMRegressor
import numpy as np
from forecaster import adjust_forecast
np.random.seed(20010524)

WindRemitDataset=pd.read_csv("data/dataset/dwd/WindRemitDataset.csv")
#提取特征归一化
features_wind=WindRemitDataset.iloc[:,:-1]
with open("data/dataset/dwd/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)
features_wind=(features_wind-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]

#原模型预测结果
Models_wind={}
for quantile in range(10,100,10):
    with open(f"models/LGBM/full/wind_q{quantile}.pkl","rb") as f:
        Models_wind[f"q{quantile}"]=pickle.load(f)
Wind_Generation_Forecast={}
for quantile in range(10,100,10):
    
    #前向
    output_wind= Models_wind[f"q{quantile}"].predict(features_wind)

    #逆归一化
    output_wind=output_wind*Dataset_stats["Std"]["labels"]["wind"]+Dataset_stats["Mean"]["labels"]["wind"]

    #负值清0
    output_wind[output_wind<0]=0

    #记录
    Wind_Generation_Forecast[f"q{quantile}"]=output_wind
    WindRemitDataset[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]

plt.plot(WindRemitDataset["Wind_MWh_credit"])
plt.plot(WindRemitDataset["q50"])

#====================================风电限电模型训练==================================

maxPower=215
#筛选出风电限电数据
MetaDataset=WindRemitDataset[WindRemitDataset[f"q{50}"]>maxPower]

#划分训练集测试集
idxs=np.random.rand(len(MetaDataset))<0.8
MetaDataset_train=MetaDataset[idxs]
MetaDataset_test=MetaDataset[~idxs]

Pinball_Losses_test={}
Pinball_Losses_full={}
for idx,quantile in enumerate(range(10,100,10)):

    #创建LGBM模型
    params={
        "num_leaves":3,
        "n_estimators":50,
        "max_depth":3,
        "lambda_l1":10,
        "lambda_l2":10,
        "objective":"quantile",
        "alpha":quantile/100,
        "verbose":-1,
        "random_state":42
    }
    model=LGBMRegressor(**params)

    #训练模型 从q50的预测结果到不同分位数的预测结果
    model.fit(np.array(MetaDataset_train[["q10","q20","q30","q40","q50","q60","q70","q80","q90"]]),
              np.array(MetaDataset_train["Wind_MWh_credit"]))
    
    #计算测试损失
    y_hat=model.predict(np.array(MetaDataset_test[["q10","q20","q30","q40","q50","q60","q70","q80","q90"]]))
    Pinball_Losses_test[f"q{quantile}"]=pinball(y=MetaDataset_test['Wind_MWh_credit'],y_hat=y_hat,alpha=quantile/100)

    #保存模型
    with open(f"models/LGBM/partial/wind_remit_q{quantile}.pkl","wb") as f:
        pickle.dump(model,f)

    #全训练
    model=LGBMRegressor(**params)

    #训练模型 从q50的预测结果到不同分位数的预测结果
    model.fit(np.array(MetaDataset[["q10","q20","q30","q40","q50","q60","q70","q80","q90"]]),
              np.array(MetaDataset["Wind_MWh_credit"]))
    #计算总损失
    y_hat=model.predict(np.array(MetaDataset[["q10","q20","q30","q40","q50","q60","q70","q80","q90"]]))
    Pinball_Losses_full[f"q{quantile}"]=pinball(y=MetaDataset['Wind_MWh_credit'],y_hat=y_hat,alpha=quantile/100)

    #保存模型
    with open(f"models/LGBM/full/wind_remit_q{quantile}.pkl","wb") as f:
        pickle.dump(model,f)


#计算平均损失
print(f"Toal Test Score:{sum(Pinball_Losses_test.values())/len(Pinball_Losses_test.values())}")
print(f"Toal Full Score:{sum(Pinball_Losses_full.values())/len(Pinball_Losses_full.values())}")


#=======================================测试=======================================
#加载模型
Models_wind_remit={}
for quantile in range(10,100,10):
    with open(f"models/LGBM/full/wind_remit_q{quantile}.pkl","rb") as f:
        Models_wind_remit[f"q{quantile}"]=pickle.load(f)

Wind_Generation_Forecast_remit={}
#筛选出限电点
idxs_limit=Wind_Generation_Forecast["q50"]>maxPower
for idx,quantile in enumerate(range(10,100,10)):
    
    Wind_Generation_Forecast_remit[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"].copy()

    #预测
    quantiles = sorted(Wind_Generation_Forecast.keys(), key=lambda x: int(x[1:]))
    forecast_array = np.array([Wind_Generation_Forecast[q] for q in quantiles]).T
    Wind_Generation_Forecast_remit[f"q{quantile}"][idxs_limit]=Models_wind_remit[f"q{quantile}"].predict(forecast_array[idxs_limit])

    #上限限制
    Wind_Generation_Forecast_remit[f"q{quantile}"][Wind_Generation_Forecast_remit[f"q{quantile}"]>maxPower]=maxPower
    
    #分位数重新排序，确保大的分位数结果更大
    Wind_Generation_Forecast_remit=adjust_forecast(Wind_Generation_Forecast_remit)

#计算损失
Pinball_Losses={}
for quantile in range(10,100,10):
    Pinball_Losses[f"q{quantile}"]=pinball(y=WindRemitDataset["Wind_MWh_credit"],y_hat=Wind_Generation_Forecast_remit[f"q{quantile}"],alpha=quantile/100)

plt.figure(figsize=(10,5))
plt.plot(Wind_Generation_Forecast_remit["q50"],label="forecast_meta")
plt.plot(Wind_Generation_Forecast["q50"],label="forecast_origin")
plt.plot(WindRemitDataset["Wind_MWh_credit"],label="real",color="red")
plt.legend()
print(f"Wind Score:{sum(Pinball_Losses.values())/len(Pinball_Losses.values())}")

plt.figure(figsize=(10,5))
for quantile in range(10,100,10):
    plt.plot(Wind_Generation_Forecast_remit[f"q{quantile}"],label=f"forecast_q{quantile}")


