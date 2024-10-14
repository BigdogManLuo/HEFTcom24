import pandas as pd
import utils
from tqdm import tqdm
import numpy as np

# Load Data
features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype="test",source="dwd")
features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype="test",source="gfs")
IntegratedDataset=pd.read_csv("data/dataset/test/IntegratedDataset.csv")

#Energy Data
energy_data=pd.read_csv("data/raw/Energy_Data_20200920_20240118.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]
energy_data["hours"]=energy_data["dtm"].dt.hour
energy_data["Price_diff"]=energy_data["DA_Price"]-energy_data["SS_Price"]
energy_data["Total_MWh"]=energy_data["Wind_MWh_credit"]+energy_data["Solar_MWh_credit"]

#Modelling table
modelling_table=energy_data[["dtm","hours","Total_MWh","SS_Price","DA_Price","Price_diff"]].reset_index(drop=True)
modelling_table=modelling_table[modelling_table["dtm"].isin(IntegratedDataset["valid_datetime"])]

#Power Forecasting Q50
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "hours":features_solar_dwd[:,-1],
    "full":False,
    "WLimit":False,
    "SolarRevise":False,
    "aggregation":True
}
Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)

modelling_table["q50"]=Total_generation_forecast["q50"]

#Power Forecasting MSE-oriented
params={
    "wind_features_dwd":features_wind_dwd,
    "wind_features_gfs":features_wind_gfs,
    "solar_features":features_solar_dwd,
    "full":False,
    "hours":modelling_table["hours"].values,
    "WLimit":False,
    "SolarRevise":False
}
modelling_table["mse"]=utils.forecast_bidding(**params).values

modelling_table["pd_pred"]=np.array([0]*len(modelling_table))


#=========================================Test Trading======================================================
for day in tqdm(pd.date_range(modelling_table["dtm"].min()-pd.Timedelta(days=64),modelling_table["dtm"].max()-pd.Timedelta(days=64))):

    start_day=day
    end_day=start_day+pd.Timedelta(days=60)
    target_day=end_day+pd.Timedelta(days=4)
    
    past_df=energy_data[(energy_data["dtm"]>=start_day) & (energy_data["dtm"]<=end_day)]
    pd_in_hours=past_df["Price_diff"].groupby(past_df["hours"]).mean()
    
    target_df=modelling_table.loc[(modelling_table["dtm"]>=target_day) & (modelling_table["dtm"]<=target_day+pd.Timedelta(days=1)-pd.Timedelta(minutes=30))]
    modelling_table.loc[(modelling_table["dtm"] >= target_day) & (modelling_table["dtm"] <= target_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30)), "pd_pred"] = pd_in_hours[target_df["hours"].values].values
    
modelling_table["q50pd"]=modelling_table["q50"]+7.14*modelling_table["pd_pred"]
modelling_table["msepd"]=modelling_table["mse"]+7.14*modelling_table["pd_pred"]
modelling_table.loc[modelling_table["q50pd"]<0,"q50pd"]=0
modelling_table.loc[modelling_table["q50pd"]>1800,"q50pd"]=1800
modelling_table.loc[modelling_table["msepd"]<0,"msepd"]=0
modelling_table.loc[modelling_table["msepd"]>1800,"msepd"]=1800

#1：q50 
R1=utils.getRevenue(modelling_table["q50"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])
RMSE_1=np.sqrt((1/len(modelling_table))*np.sum((modelling_table["q50"]-modelling_table["Total_MWh"])**2))
MAE_1=(1/len(modelling_table))*np.sum(np.abs(modelling_table["q50"]-modelling_table["Total_MWh"]))

#2：mse
R2=utils.getRevenue(modelling_table["mse"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])
RMSE_2=np.sqrt((1/len(modelling_table))*np.sum((modelling_table["mse"]-modelling_table["Total_MWh"])**2))
MAE_2=(1/len(modelling_table))*np.sum(np.abs(modelling_table["mse"]-modelling_table["Total_MWh"]))

#3: q50+ST
R3=utils.getRevenue(modelling_table["q50pd"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])
RMSE_3=np.sqrt((1/len(modelling_table))*np.sum((modelling_table["q50pd"]-modelling_table["Total_MWh"])**2))
MAE_3=(1/len(modelling_table))*np.sum(np.abs(modelling_table["q50pd"]-modelling_table["Total_MWh"]))

#4: mse+ST
R4=utils.getRevenue(modelling_table["msepd"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])
RMSE_4=np.sqrt((1/len(modelling_table))*np.sum((modelling_table["msepd"]-modelling_table["Total_MWh"])**2))
MAE_4=(1/len(modelling_table))*np.sum(np.abs(modelling_table["msepd"]-modelling_table["Total_MWh"]))

print("\n")
print(f"Strategy 1: {R1.sum()}, RMSE: {RMSE_1}, MAE: {MAE_1}")
print(f"Strategy 2: {R2.sum()}, RMSE: {RMSE_2}, MAE: {MAE_2}","Improvement: ",np.round(100*((R2.sum()-R1.sum())/R1.sum()),2))
print(f"Strategy 3: {R3.sum()}, RMSE: {RMSE_3}, MAE: {MAE_3}", "Improvement: ",np.round(100*((R3.sum()-R1.sum())/R1.sum()),2))
print(f"Strategy 4: {R4.sum()}, RMSE: {RMSE_4}, MAE: {MAE_4}", "Improvement: ",np.round(100*((R4.sum()-R1.sum())/R1.sum()),2))
