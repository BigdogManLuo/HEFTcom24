import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import comp_utils
from comp_utils import pinball
import forecaster


#读取Integrated Dataset
IntegratedDataset=pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")

#=======================================高风电功率时段的特征分布=======================================
#取IntegratedDataset的30%-80%的数据
A=IntegratedDataset.iloc[int(len(IntegratedDataset)*0.3):int(len(IntegratedDataset)*0.8)].reset_index(drop=True)

#挑选出Wind_MWh_credit大于500的数据
A=A[A["Wind_MWh_credit"]>500].reset_index(drop=True)

#查看其中ws_100_t+1_dwd_1这一列数据的分布范围
print(A["ws_100_t_dwd_1"].describe())

#可视化
plt.figure(figsize=(8,6))
plt.hist(A["ws_100_t_dwd_1"],bins=100)
plt.title("ws_100_t_dwd_1")
plt.show()

#=======================================低风电功率时段的特征分布=======================================
#取IntegratedDataset的30%-80%的数据
B=IntegratedDataset.iloc[int(len(IntegratedDataset)*0.3):int(len(IntegratedDataset)*0.8)].reset_index(drop=True)

#挑选出Wind_MWh_credit大于500的数据
B=B[B["Wind_MWh_credit"]<200].reset_index(drop=True)

#查看其中ws_100_t+1_dwd_1这一列数据的分布范围
print(B["ws_100_t_dwd_1"].describe())

#可视化
plt.figure(figsize=(8,6))
plt.hist(B["ws_100_t_dwd_1"],bins=100)
plt.title("ws_100_t_dwd_1")
plt.show()

#=======================================高风速时段的风电功率分布=======================================
B=IntegratedDataset.iloc[int(len(IntegratedDataset)*0.3):int(len(IntegratedDataset)*0.99)].reset_index(drop=True)

#挑选出ws_100_t_dwd_1>20的数据
B=B[B["ws_100_t+1_dwd_1"]>30].reset_index(drop=True)

#查看Wind_MWh_credit的分布范围
print(B["Wind_MWh_credit"].describe())

#可视化
plt.figure(figsize=(8,6))
plt.hist(B["Wind_MWh_credit"],bins=100)
plt.title("Wind_MWh_credit")
plt.show()

plt.figure(figsize=(8,6))
plt.hist(B["ws_100_t+1_dwd_max"]-B["ws_100_t+1_dwd_min"],bins=100)
plt.show()


#=======================================高风速时段的风电功率分布=======================================
C=IntegratedDataset.iloc[int(len(IntegratedDataset)*0.3):int(len(IntegratedDataset)*0.99)].reset_index(drop=True)

#挑选出ws_100_t_dwd_1>20 且 Wind_MWh_credit<350的数据
C=C[(C["ws_100_t+1_dwd_1"]>22) & (C["Wind_MWh_credit"]<400)].reset_index(drop=True)


#可视化
plt.figure(figsize=(8,6))
plt.hist(C["ws_100_t+1_dwd_max"]-C["ws_100_t+1_dwd_min"],bins=100)
