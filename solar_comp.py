import pandas as pd
import utils
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

#Forecasting
modelling_table=pd.read_csv("data/dataset/latest/dwd/SolarDataset.csv")
modelling_table=utils.forecastWithoutPostProcessing(modelling_table)


#Scatter plot
plt.figure(figsize=(8,6))
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams.update({'font.size': 21})
s1=plt.scatter(modelling_table["Solar_MWh_credit"],modelling_table["q50"],s=6,color="#0984e3")
l1=plt.plot([0,800],[0,800],color="#d63031",linestyle="--")
lr=LinearRegression()
lr.fit(modelling_table["Solar_MWh_credit"].values.reshape(-1,1),modelling_table["q50"].values)
x_range=np.arange(0,800)
y_range=lr.predict(x_range.reshape(-1,1))
l2=plt.plot(x_range,y_range,color="black",linestyle="--")

plt.legend([l1[0],l2[0]],["y=x","LR fit"],frameon=False,loc="upper left")

plt.xlim(0,900)
plt.ylim(0,900)

plt.xlabel("True Solar Generation (MWh)")
plt.ylabel("Predicted Solar Generation (MWh)")
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tick_params(width=2)
plt.tight_layout()
plt.savefig("figs/solar_scatter.png",dpi=660)