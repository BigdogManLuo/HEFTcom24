import pandas as pd
import numpy as np
import xarray as xr

#%% 处理气象数据

#加载原始数据
demand_old=xr.open_dataset("data/dwd_icon_eu_demand_20200920_20231027.nc").to_dataframe().reset_index()
demand_new=xr.open_dataset("data/dwd_icon_eu_demand_20231027_20240108.nc").to_dataframe().reset_index()
demand_latest=xr.open_dataset("data/dwd_icon_eu_demand_20240108_20240129.nc").to_dataframe().reset_index()
demand_new.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
demand_latest.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

#合并
demand_features=pd.concat([demand_old,demand_new,demand_latest])


#获取每个坐标的气象数据
demand_features['point'] = 'p' + demand_features['point'].astype(str)
pivot_columns = ['RelativeHumidity', 'Temperature', 'TotalPrecipitation', 'WindSpeed']

demand_features = demand_features.pivot_table(index=['ref_datetime', 'valid_datetime'],
                                columns='point',
                                values=pivot_columns,
                                aggfunc='mean').reset_index()
demand_features.columns = ['_'.join(col).strip() if col[1] else col[0] for col in demand_features.columns.values]


#标准化时间
demand_features["ref_datetime"] = demand_features["ref_datetime"].dt.tz_localize("UTC")
demand_features["valid_datetime"] = demand_features["ref_datetime"] + pd.TimedeltaIndex(demand_features["valid_datetime"],unit="hours")


#删除非最新的预报数据
demand_features = demand_features[demand_features["valid_datetime"] - demand_features["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)


#%% 处理电价数据
energy_data = pd.read_csv("data/Energy_Data_20200920_20240118.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])

#提取日前市场和平衡市场价格数据
energy_data = energy_data[["dtm","SS_Price","DA_Price"]]


#%% 构造数据集

#电价数据合并到气象数据
modelling_table = demand_features.merge(energy_data,how="left",left_on="valid_datetime",right_on="dtm")

#缺失值丢弃
modelling_table = modelling_table.dropna()

#特征工程
modelling_table["hour"] = modelling_table["valid_datetime"].dt.hour
modelling_table["is_weekend"]=modelling_table["valid_datetime"].dt.weekday.isin([5,6]).astype(int)

#导出
modelling_table.drop(columns=["ref_datetime","valid_datetime","dtm"],inplace=True)
modelling_table.to_csv("data/dataset/price_modelling_table.csv",index=False)


