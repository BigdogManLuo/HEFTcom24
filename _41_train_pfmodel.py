import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

#加载数据
modelling_table = pd.read_csv("data/dataset/price_modelling_table.csv")
X=modelling_table.drop(columns=["SS_Price","DA_Price"])
y=modelling_table[["SS_Price","DA_Price"]]

#分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#训练模型（lightGBM）
model = lgb.LGBMRegressor(n_estimators=1000)
model.fit(X_train, y_train["SS_Price"])

#测试
y_pred = model.predict(X_test)
print(mean_squared_error(y_test["SS_Price"], y_pred))

#可视化
plt.scatter(y_test["SS_Price"], y_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()