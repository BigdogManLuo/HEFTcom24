import pandas as pd
import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from comp_utils import pinball
import pickle

#%% -------------------------训练--------------------------------

#读取数据集
WindDataset=pd.read_csv("data/dataset/WindDataset.csv")

#提取features和labels
features=WindDataset.iloc[:,:-1]
labels=WindDataset.iloc[:,-1]


#z-score标准化
with open("data/dataset/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

features=(features-Dataset_stats["Mean"]["features"]["wind"])/Dataset_stats["Std"]["features"]["wind"]
labels=(labels-Dataset_stats["Mean"]["labels"]["wind"])/Dataset_stats["Std"]["labels"]["wind"]

#转换为numpy数组
features=np.array(features)
labels=np.array(labels)

#划分训练集、验证集和测试集7:1:2
train_features=features[:int(0.7*len(features))]
train_labels=labels[:int(0.7*len(labels))]
val_features=features[int(0.7*len(features)):int(0.8*len(features))]
val_labels=labels[int(0.7*len(labels)):int(0.8*len(labels))]
test_features=features[int(0.8*len(features)):]
test_labels=labels[int(0.8*len(labels)):]


#创建训练集、验证集和测试集
train_data=lgb.Dataset(train_features,label=train_labels)
valid_data=lgb.Dataset(val_features,label=val_labels)
test_data=lgb.Dataset(test_features,label=test_labels)
full_data=lgb.Dataset(features,label=labels)


#超参数
num_round = 10
param = {'num_leaves':25,
        'num_trees':100,
        'min_data_in_leaf':50,
        'objective':'quantile',
        } 

#选择是否全训练
_input=input("是否全训练？(y/n)")
if _input=="y":
    full=True
elif _input=="n":
    full=False
else:
    raise ValueError("输入错误")

if full:
    #训练
    LGBM_models={}
    for quantile in range(10,100,10):
        param['alpha']=quantile/100
        bst = lgb.train(param, full_data, num_round, valid_sets=[valid_data])
        LGBM_models[f"q{quantile}"]=bst

    #保存模型
    for quantile in range(10,100,10):
        LGBM_models[f"q{quantile}"].save_model(f"models/LGBM/full/wind_q{quantile}.txt")

else:
    #训练
    LGBM_models={}
    for quantile in range(10,100,10):
        param['alpha']=quantile/100
        bst = lgb.train(param, train_data, num_round, valid_sets=[valid_data])
        LGBM_models[f"q{quantile}"]=bst

    #保存模型
    for quantile in range(10,100,10):
        LGBM_models[f"q{quantile}"].save_model(f"models/LGBM/partial/wind_q{quantile}.txt")



#%% ------------------------测试--------------------------------

#加载模型
LGBM_models={}
for quantile in range(10,100,10):
    bst = lgb.Booster(model_file=f"models/LGBM/partial/wind_q{quantile}.txt")
    LGBM_models[f"q{quantile}"]=bst

#加载数据集
Labels_test=np.array(WindDataset.iloc[:,-1])[int(0.8*len(labels)):]

#预测
Wind_Generation_forecast={}
Pinball_Loss={}
for quantile in range(10,100,10):

    #前向
    y_hat=LGBM_models[f"q{quantile}"].predict(test_features)

    #反归一化
    y_hat=y_hat*Dataset_stats["Std"]["labels"]["wind"]+Dataset_stats["Mean"]["labels"]["wind"]

    #计算损失
    loss=pinball(y=Labels_test,y_hat=y_hat,alpha=quantile/100)

    #保存
    Wind_Generation_forecast[f"q{quantile}"]=y_hat
    Pinball_Loss[f"q{quantile}"]=loss


#计算平均得分
Pinball_Loss_mean=np.array(list(Pinball_Loss.values())).mean()
print("Score:",Pinball_Loss_mean)

    

#绘制预测值和真实值的散点图
plt.scatter(Labels_test,Wind_Generation_forecast["q50"],color="blue",s=10,alpha=0.3)
plt.plot(Labels_test,Labels_test,color="red",linestyle="--")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.show()


