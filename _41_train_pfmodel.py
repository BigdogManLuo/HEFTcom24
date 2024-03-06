import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import NNRegressor

#加载数据
modelling_table = pd.read_csv("data/dataset/price_modelling_table.csv")
X=modelling_table.drop(columns=["SS_Price","DA_Price"])
y=modelling_table[["SS_Price","DA_Price"]]

#分为训练集、验证集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


#初始化模型  lrbatch_size,num_epochs,input_size,output_size,hidden_size,loss_fn,alpha,num_layers
params = {
   "lr": 0.01,
    "batch_size": 256,
    "num_epochs": 100,
    "input_size": X_train.shape[1],
    "output_size": y_train.shape[1],
    "hidden_size": 64,
    "loss_fn": "mse",
}



#可视化
plt.scatter(y_test["SS_Price"], y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()