import pandas as pd
import numpy as np

#读取Integrated Dataset
IntegratedDataset = pd.read_csv("data/dataset/dwd/IntegratedDataset.csv")

#取出Integrated Dataset中的ref_datetime为00:00时刻的数据
IntegratedDataset["ref_datetime"] = pd.to_datetime(IntegratedDataset["ref_datetime"])
IntegratedDataset["valid_datetime"] = pd.to_datetime(IntegratedDataset["valid_datetime"])
IntegratedDataset=IntegratedDataset[IntegratedDataset["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)

#取出Integrated Dataset中的valid_datetime与ref_datetime的差值为23小时~47小时的数据
IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])<=np.timedelta64(47,"h")]
IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])>=np.timedelta64(23,"h")]

#取后15%作为测试集
IntegratedDataset = IntegratedDataset.iloc[int(len(IntegratedDataset)*0.85):,:].reset_index(drop=True)

#风电光伏数据集
columns_wind=pd.read_csv("data/dataset/dwd/WindDataset.csv").columns.tolist()
columns_solar=pd.read_csv("data/dataset/dwd/SolarDataset.csv").columns.tolist()

WindDataset=IntegratedDataset[columns_wind]
SolarDataset=IntegratedDataset[columns_solar]

#%% 保存数据集
IntegratedDataset.to_csv('data/dataset/test/IntegratedDataset.csv',index=False)
WindDataset.to_csv('data/dataset/test/WindDataset.csv',index=False)
SolarDataset.to_csv('data/dataset/test/SolarDataset.csv',index=False)

print("done!")