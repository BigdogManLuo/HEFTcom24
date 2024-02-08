import pandas as pd
import lightgbm as lgb
import numpy as np

#创建训练集
X_train=np.random.rand(1000,10)
y_train=X_train.sum(axis=1)+np.random.rand(1000,)
train_data=lgb.Dataset(X_train,label=y_train)

#创建验证集
X_val=np.random.rand(1000,10)
y_val=X_val.sum(axis=1)+np.random.rand(1000,)
val_data=lgb.Dataset(X_val,label=y_val)

#超参数
num_round = 10
param = {'num_leaves':30,
        'num_trees':100,
        'min_data_in_leaf':20,
        } 
param['metric'] = 'mse'

#训练
bst = lgb.train(param, train_data, num_round, valid_sets=[val_data])

#保存模型
bst.save_model('model.txt')

#加载模型
bst = lgb.Booster(model_file='model.txt')  #init model

#测试
y_vhat=bst.predict(X_val)

#计算y_vhat和y_val的MAE
mae=np.abs(y_vhat-y_val).mean()

#绘制预测值和真实值的散点图
import matplotlib.pyplot as plt
plt.scatter(y_val,y_vhat)
plt.show()

#计算相关性
np.corrcoef(y_val,y_vhat)

#计算R2
from sklearn.metrics import r2_score
print(f"R2:{r2_score(y_val,y_vhat)}")
