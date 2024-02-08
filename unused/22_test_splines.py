from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from comp_utils import pinball
import pickle

#加载数据集
WindDataset=pd.read_csv("data/dataset/WindDataset.csv")
SolarDataset=pd.read_csv("data/dataset/SolarDataset.csv")

#修改列名
WindDataset.columns=["ws_100_t_dwd_1","ws_100_t_dwd_2","ws_100_t_dwd_3","ws_100_t1_dwd_1","ws_100_t1_dwd_2","ws_100_t1_dwd_3","Wind_MWh_credit"]
SolarDataset.columns=["rad_1t_dwd","rad_t_dwd","rad_t1_dwd","Solar_MWh_credit"]

#加载原风电、光伏数据集的均值和方差
Mean_features_wind=np.load("data/dataset/Mean_features_wind.npy")
Std_features_wind=np.load("data/dataset/Std_features_wind.npy")
Mean_features_solar=np.load("data/dataset/Mean_features_solar.npy")
Std_features_solar=np.load("data/dataset/Std_features_solar.npy")


#z-score标准化(仅对特征)
WindDataset.iloc[:,:-1]=(WindDataset.iloc[:,:-1]-Mean_features_wind)/Std_features_wind
SolarDataset.iloc[:,:-1]=(SolarDataset.iloc[:,:-1]-Mean_features_solar)/Std_features_solar

#划分训练集、测试集
train_dataset_wind=WindDataset.iloc[:int(0.9*len(WindDataset))]
test_dataset_wind=WindDataset.iloc[int(0.1*len(WindDataset)):]

train_dataset_solar=SolarDataset.iloc[:int(0.9*len(SolarDataset))]
test_dataset_solar=SolarDataset.iloc[int(0.1*len(SolarDataset)):]

#加载分位数回归模型
with open("models/splines/SPL_wind.pkl","rb") as f:
    models_wind=pickle.load(f)

with open("models/splines/SPL_solar.pkl","rb") as f:
    models_solar=pickle.load(f)

#测试预测效果
output_wind=dict()
output_solar=dict()
pinball_losses=dict()

for quantile in range(10,100,10):
    #测试
    output_wind[f"q{quantile}"] = models_wind[f"q{quantile}"].predict(test_dataset_wind)
    output_solar[f"q{quantile}"] = models_solar[f"q{quantile}"].predict(test_dataset_solar)
    
    output=output_wind[f"q{quantile}"]+output_solar[f"q{quantile}"]

    #手动修正
    output[output<0]=0

    #计算损失
    pinball_losses[f"q{quantile}"] = pinball(y_hat=output, 
                                             y=np.array(test_dataset_wind["Wind_MWh_credit"]+test_dataset_solar["Solar_MWh_credit"]), 
                                             alpha=quantile/100)

#打印平均损失
print(np.mean(list(pinball_losses.values())))

