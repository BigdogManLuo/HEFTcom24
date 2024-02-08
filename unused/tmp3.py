import pandas as pd

#%% 查看夜晚时刻的光伏发电数据
#读取Integrated Dataset
IntegratedDataset=pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")

#将valid_datetime这一列变为只有小时的列
IntegratedDataset["valid_datetime"]=pd.to_datetime(IntegratedDataset["valid_datetime"])
IntegratedDataset["valid_datetime"]=IntegratedDataset["valid_datetime"].dt.hour

#按valid_datetime进行分组，并展示每组的"Solar_MWh_credit"的分布
print(IntegratedDataset.groupby("valid_datetime")["Solar_MWh_credit"].describe())

