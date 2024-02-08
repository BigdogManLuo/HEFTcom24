from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from comp_utils import pinball

#加载数据集
WindDataset=pd.read_csv("data/dataset/WindDataset.csv")

#修改列名
WindDataset.columns=["ws_100_t_dwd_1","ws_100_t_dwd_2","ws_100_t_dwd_3","ws_100_t1_dwd_1","ws_100_t1_dwd_2","ws_100_t1_dwd_3","Wind_MWh_credit"]

#加载原风电、光伏数据集的均值和方差
Mean_features_wind=np.load("data/dataset/Mean_features_wind.npy")
Std_features_wind=np.load("data/dataset/Std_features_wind.npy")

#z-score标准化(仅对特征)
WindDataset.iloc[:,:-1]=(WindDataset.iloc[:,:-1]-Mean_features_wind)/Std_features_wind

#划分训练集、测试集
train_dataset=WindDataset.iloc[:int(0.9*len(WindDataset))]
test_dataset=WindDataset.iloc[int(0.1*len(WindDataset)):]


#分位数回归模型
mod = smf.quantreg('Wind_MWh_credit ~ bs(ws_100_t_dwd_1, df=3) + bs(ws_100_t_dwd_2, df=3)+ bs(ws_100_t_dwd_3, df=3)+bs(ws_100_t1_dwd_1, df=3) + bs(ws_100_t1_dwd_2, df=3)+ bs(ws_100_t1_dwd_3, df=3)',
                   data=train_dataset)

models = dict()
pinball_losses = dict()
output=dict()

#分位数回归
for quantile in range(10,100,10):
    
    #拟合
    models[f"q{quantile}"] = mod.fit(q=quantile/100,max_iter=10000)
    
    #测试
    output[f"q{quantile}"] = models[f"q{quantile}"].predict(test_dataset)
    #output[f"q{quantile}"][output[f"q{quantile}"]<0, f"q{quantile}"] = 0
    
    #计算损失
    pinball_losses[f"q{quantile}"] = pinball(y_hat=output[f"q{quantile}"], 
                                             y=np.array(test_dataset["Wind_MWh_credit"]), 
                                             alpha=quantile/100)


#打印平均损失
print(np.mean(list(pinball_losses.values())))

'''
#%% 在所有数据集上训练
mod = smf.quantreg('Wind_MWh_credit ~ bs(ws_100_t_dwd_1, df=3) + bs(ws_100_t_dwd_2, df=3)+ bs(ws_100_t_dwd_3, df=3)+bs(ws_100_t1_dwd_1, df=3) + bs(ws_100_t1_dwd_2, df=3)+ bs(ws_100_t1_dwd_3, df=3)',
                   data=WindDataset)
models = dict()

#分位数回归
for quantile in range(10,100,10):
    
    #拟合
    models[f"q{quantile}"] = mod.fit(q=quantile/100,max_iter=10000)
'''

#%% 保存模型
import pickle
with open("models/splines/SPL_wind.pkl","wb") as f:
    pickle.dump(models,f)

