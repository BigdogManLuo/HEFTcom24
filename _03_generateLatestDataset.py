import pandas as pd
import xarray as xr
import utils_dp
import os

# Load DWD Wind Data
dwd_Hornsea1 = xr.open_dataset("data/raw/dwd_icon_eu_hornsea_1_20240129_20240519.nc")
dwd_Hornsea1_features=utils_dp.preProcessNWPData(dwd_Hornsea1,featureType="wind",featureName=["WindSpeed:100"])

# Load DWD Solar Data
dwd_solar=xr.open_dataset("data/raw/dwd_icon_eu_pes10_20240129_20240519.nc")
dwd_solar_features=utils_dp.preProcessNWPData(dwd_solar,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

#Load GFS Wind Data
gfs_Hornsea1 = xr.open_dataset("data/raw/ncep_gfs_hornsea_1_20240129_20240519.nc")
gfs_Hornsea1_features=utils_dp.preProcessNWPData(gfs_Hornsea1,featureType="wind",featureName=["WindSpeed:100"])

#Load GFS Solar Data
gfs_solar=xr.open_dataset("data/raw/ncep_gfs_pes10_20240129_20240519.nc")
gfs_solar_features=utils_dp.preProcessNWPData(gfs_solar,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

# Load Energy Data
energy_data = pd.read_csv("data/raw/Energy_Data_20240119_20240519.csv")
energy_data = utils_dp.preProcessEnergyData(energy_data)

# Generate Integrated Dataset
IntegratedDataset_dwd=utils_dp.generateIntegratedDataset(dwd_Hornsea1_features,dwd_solar_features,energy_data,"dwd")
IntegratedDataset_gfs=utils_dp.generateIntegratedDataset(gfs_Hornsea1_features,gfs_solar_features,energy_data,"gfs")


# Generate Testset
IntegratedDataset_dwd=utils_dp.generateTestset(IntegratedDataset_dwd,start_time="2024-02-19 23:00",end_time="2024-05-19 22:30").reset_index(drop=True)
IntegratedDataset_gfs=utils_dp.generateTestset(IntegratedDataset_gfs,start_time="2024-02-19 23:00",end_time="2024-05-19 22:30").reset_index(drop=True)

#Filling in missing values
time_series=pd.date_range(start="2024-02-19 23:00",end="2024-05-19 22:30",freq="30T",tz="UTC")
IntegratedDataset_dwd = pd.merge(pd.DataFrame({'valid_datetime': time_series}), IntegratedDataset_dwd, on='valid_datetime', how='left')
df_supplem1=pd.read_csv("data/2024-05-04_Supplem.csv")
df_supplem2=pd.read_csv("data/2024-05-13_Supplem.csv")
#Merge
df_supplem1["valid_datetime"]=pd.to_datetime(df_supplem1["valid_datetime"]).dt.tz_convert("UTC")
df_supplem2["valid_datetime"]=pd.to_datetime(df_supplem2["valid_datetime"]).dt.tz_convert("UTC")
energy_data["dtm"]=pd.to_datetime(energy_data["dtm"]).dt.tz_convert("UTC")
IntegratedDataset_dwd.set_index("valid_datetime", inplace=True)
df_supplem1.set_index("valid_datetime", inplace=True)
df_supplem2.set_index("valid_datetime", inplace=True)
energy_data.rename(columns={"dtm": "valid_datetime"}, inplace=True)
energy_data.set_index("valid_datetime", inplace=True)
IntegratedDataset_dwd = IntegratedDataset_dwd.combine_first(df_supplem1)
IntegratedDataset_dwd = IntegratedDataset_dwd.combine_first(df_supplem2)
IntegratedDataset_dwd = IntegratedDataset_dwd.combine_first(energy_data)
IntegratedDataset_dwd=IntegratedDataset_dwd.dropna().reset_index()


# Align GFS with DWD in timestamp, missing values in GFS are filled with DWD
IntegratedDataset_gfs = pd.merge(pd.DataFrame({'valid_datetime': IntegratedDataset_dwd['valid_datetime']}), IntegratedDataset_gfs, on='valid_datetime', how='left')
IntegratedDataset_gfs.rename(columns=lambda x: x.replace('gfs', 'dwd') if 'gfs' in x else x, inplace=True)
IntegratedDataset_gfs = IntegratedDataset_gfs.combine_first(IntegratedDataset_dwd) 
IntegratedDataset_gfs.rename(columns=lambda x: x.replace('dwd', 'gfs') if 'dwd' in x else x, inplace=True)

# Merge Integrated Dataset
IntegratedDataset=IntegratedDataset_dwd.merge(IntegratedDataset_gfs,how="inner",on=["valid_datetime"])# Merge Integrated Dataset
IntegratedDataset.rename(columns={"Wind_MWh_credit_x":"Wind_MWh_credit","Solar_MWh_credit_x":"Solar_MWh_credit","total_generation_MWh_x":"total_generation_MWh","hours_x":"hours","DA_Price_x":"DA_Price","SS_Price_x":"SS_Price"},inplace=True)

# Available Capacity according to the Remit information
availableCapacity=pd.Series([415]*len(IntegratedDataset))
availableCapacity.iloc[IntegratedDataset["valid_datetime"]<"2024-02-24 13:00"]=225
availableCapacity.iloc[(IntegratedDataset["valid_datetime"]>="2024-02-24 13:00") & (IntegratedDataset["valid_datetime"]<"2024-02-25 22:00")]=265
availableCapacity.iloc[(IntegratedDataset["valid_datetime"]>="2024-02-25 22:00") & (IntegratedDataset["valid_datetime"]<"2024-02-26 01:00")]=315
availableCapacity.iloc[2168:3140]=400
availableCapacity.iloc[3140:3749]=380
availableCapacity.iloc[4150:]=215
IntegratedDataset["availableCapacity"]=availableCapacity

# Generate Wind Solar Dataset
WindDataset_dwd,SolarDataset_dwd=utils_dp.generateWindSolarDataset(IntegratedDataset,source="dwd")
WindDataset_gfs,SolarDataset_gfs=utils_dp.generateWindSolarDataset(IntegratedDataset,source="gfs")

# Save Dataset
if not os.path.exists("data/dataset/latest/dwd"):
    os.makedirs("data/dataset/latest/dwd")
if not os.path.exists("data/dataset/latest/gfs"):
    os.makedirs("data/dataset/latest/gfs")
    
IntegratedDataset.to_csv('data/dataset/latest/IntegratedDataset.csv',index=False)
WindDataset_dwd.to_csv('data/dataset/latest/dwd/WindDataset.csv',index=False)
SolarDataset_dwd.to_csv('data/dataset/latest/dwd/SolarDataset.csv',index=False)
WindDataset_gfs.to_csv('data/dataset/latest/gfs/WindDataset.csv',index=False)
SolarDataset_gfs.to_csv('data/dataset/latest/gfs/SolarDataset.csv',index=False)
