import xarray as xr
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils_data import *

def generateFullDataset():
    
    # Load DWD Wind Data
    dwd_Hornsea1_old = xr.open_dataset("../data/raw/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
    dwd_Hornsea1_new = xr.open_dataset("../data/raw/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
    dwd_Hornsea1_latest = xr.open_dataset("../data/raw/dwd_icon_eu_hornsea_1_20240108_20240129.nc")

    dwd_Hornsea1_features_old=preProcessNWPData(dwd_Hornsea1_old,featureType="wind",featureName=["WindSpeed:100"])
    dwd_Hornsea1_features_new=preProcessNWPData(dwd_Hornsea1_new,featureType="wind",featureName=["WindSpeed:100"])
    dwd_Hornsea1_features_latest=preProcessNWPData(dwd_Hornsea1_latest,featureType="wind",featureName=["WindSpeed:100"])

    dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    dwd_Hornsea1_features_new = dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new,dwd_Hornsea1_features_latest],axis=0).reset_index(drop=True)

    # Load DWD Solar Data
    dwd_solar_old=xr.open_dataset("../data/raw/dwd_icon_eu_pes10_20200920_20231027.nc")
    dwd_solar_new=xr.open_dataset("../data/raw/dwd_icon_eu_pes10_20231027_20240108.nc")
    dwd_solar_latest=xr.open_dataset("../data/raw/dwd_icon_eu_pes10_20240108_20240129.nc")

    dwd_solar_features_old=preProcessNWPData(dwd_solar_old,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    dwd_solar_features_new=preProcessNWPData(dwd_solar_new,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    dwd_solar_features_latest=preProcessNWPData(dwd_solar_latest,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    dwd_solar_features_new = dwd_solar_features_new[dwd_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new,dwd_solar_features_latest],axis=0).reset_index(drop=True)

    # Load GFS Wind Data
    gfs_Hornsea1_old = xr.open_dataset("../data/raw/ncep_gfs_hornsea_1_20200920_20231027.nc")
    gfs_Hornsea1_new = xr.open_dataset("../data/raw/ncep_gfs_hornsea_1_20231027_20240108.nc")
    gfs_Hornsea1_latest = xr.open_dataset("../data/raw/ncep_gfs_hornsea_1_20240108_20240129.nc")

    gfs_Hornsea1_features_old=preProcessNWPData(gfs_Hornsea1_old,featureType="wind",featureName=["WindSpeed:100"])
    gfs_Hornsea1_features_new=preProcessNWPData(gfs_Hornsea1_new,featureType="wind",featureName=["WindSpeed:100"])
    gfs_Hornsea1_features_latest=preProcessNWPData(gfs_Hornsea1_latest,featureType="wind",featureName=["WindSpeed:100"])

    gfs_Hornsea1_features_old = gfs_Hornsea1_features_old[gfs_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    gfs_Hornsea1_features_new = gfs_Hornsea1_features_new[gfs_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    gfs_Hornsea1_features=pd.concat([gfs_Hornsea1_features_old,gfs_Hornsea1_features_new,gfs_Hornsea1_features_latest],axis=0).reset_index(drop=True)

    # Load GFS Solar Data
    gfs_solar_old=xr.open_dataset("../data/raw/ncep_gfs_pes10_20200920_20231027.nc")
    gfs_solar_new=xr.open_dataset("../data/raw/ncep_gfs_pes10_20231027_20240108.nc")
    gfs_solar_latest=xr.open_dataset("../data/raw/ncep_gfs_pes10_20240108_20240129.nc")

    gfs_solar_features_old=preProcessNWPData(gfs_solar_old,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    gfs_solar_features_new=preProcessNWPData(gfs_solar_new,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    gfs_solar_features_latest=preProcessNWPData(gfs_solar_latest,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    gfs_solar_features_old = gfs_solar_features_old[gfs_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    gfs_solar_features_new = gfs_solar_features_new[gfs_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    gfs_solar_features=pd.concat([gfs_solar_features_old,gfs_solar_features_new,gfs_solar_features_latest],axis=0).reset_index(drop=True)

    # Load Energy Data
    energy_data = pd.read_csv("../data/raw/Energy_Data_20200920_20240118.csv")
    energy_data = preProcessEnergyData(energy_data)

    # Generate Integrated Dataset
    IntegratedDataset_dwd=generateIntegratedDataset(dwd_Hornsea1_features,dwd_solar_features,energy_data,"dwd")
    IntegratedDataset_gfs=generateIntegratedDataset(gfs_Hornsea1_features,gfs_solar_features,energy_data,"gfs")

    #Generate Wind and Solar Dataset
    WindDataset_dwd,SolarDataset_dwd=generateWindSolarDataset(IntegratedDataset_dwd,"dwd")
    WindDataset_gfs,SolarDataset_gfs=generateWindSolarDataset(IntegratedDataset_gfs,"gfs")

    #Save Dataset
    if not os.path.exists("../data/dataset/full/dwd"):
        os.makedirs("../data/dataset/full/dwd")
    IntegratedDataset_dwd.to_csv('../data/dataset/full/dwd/IntegratedDataset.csv',index=False)
    WindDataset_dwd.to_csv('../data/dataset/full/dwd/WindDataset.csv',index=False)
    SolarDataset_dwd.to_csv('../data/dataset/full/dwd/SolarDataset.csv',index=False)

    if not os.path.exists("../data/dataset/full/gfs"):
        os.makedirs("../data/dataset/full/gfs")
    IntegratedDataset_gfs.to_csv('../data/dataset/full/gfs/IntegratedDataset.csv',index=False)
    WindDataset_gfs.to_csv('../data/dataset/full/gfs/WindDataset.csv',index=False)
    SolarDataset_gfs.to_csv('../data/dataset/full/gfs/SolarDataset.csv',index=False)

def generateTrainTestDataset():
    
    #Load Datasets
    IntegratedDataset_dwd = pd.read_csv("../data/dataset/full/dwd/IntegratedDataset.csv")    
    IntegratedDataset_gfs = pd.read_csv("../data/dataset/full/gfs/IntegratedDataset.csv")    

    #Merge Datasets
    IntegratedDataset=IntegratedDataset_gfs.merge(IntegratedDataset_dwd,how="inner",on=["ref_datetime","valid_datetime"])
    IntegratedDataset.rename(columns={"Wind_MWh_credit_x":"Wind_MWh_credit","Solar_MWh_credit_x":"Solar_MWh_credit","total_generation_MWh_x":"total_generation_MWh","hours_x":"hours","DA_Price_x":"DA_Price","SS_Price_x":"SS_Price"},inplace=True)

    #Train Test Split
    start_datetime = "2023-02-01"
    end_datetime = "2023-08-01"
    IntegratedDataset_test = generateTestset(IntegratedDataset,start_datetime,end_datetime)
    IntegratedDataset_train = generateTrainset(IntegratedDataset,IntegratedDataset_test)

    WindDataset_train_dwd,SolarDataset_train_dwd=generateWindSolarDataset(IntegratedDataset_train,"dwd")
    WindDataset_train_gfs,SolarDataset_train_gfs=generateWindSolarDataset(IntegratedDataset_train,"gfs")
    WindDataset_test_dwd,SolarDataset_test_dwd=generateWindSolarDataset(IntegratedDataset_test,"dwd")
    WindDataset_test_gfs,SolarDataset_test_gfs=generateWindSolarDataset(IntegratedDataset_test,"gfs")

    #Save Datasets
    if not os.path.exists("../data/dataset/train/dwd"):
        os.makedirs("../data/dataset/train/dwd")
    if not os.path.exists("../data/dataset/train/gfs"):
        os.makedirs("../data/dataset/train/gfs")
    if not os.path.exists("../data/dataset/test/dwd"):
        os.makedirs("../data/dataset/test/dwd")
    if not os.path.exists("../data/dataset/test/gfs"):
        os.makedirs("../data/dataset/test/gfs")

    WindDataset_train_dwd.to_csv("../data/dataset/train/dwd/WindDataset.csv",index=False)
    SolarDataset_train_dwd.to_csv("../data/dataset/train/dwd/SolarDataset.csv",index=False)
    IntegratedDataset_train.to_csv("../data/dataset/train/IntegratedDataset.csv",index=False)
    WindDataset_train_gfs.to_csv("../data/dataset/train/gfs/WindDataset.csv",index=False)
    SolarDataset_train_gfs.to_csv("../data/dataset/train/gfs/SolarDataset.csv",index=False)
    WindDataset_test_dwd.to_csv("../data/dataset/test/dwd/WindDataset.csv",index=False)
    SolarDataset_test_dwd.to_csv("../data/dataset/test/dwd/SolarDataset.csv",index=False)
    WindDataset_test_gfs.to_csv("../data/dataset/test/gfs/WindDataset.csv",index=False)
    SolarDataset_test_gfs.to_csv("../data/dataset/test/gfs/SolarDataset.csv",index=False)
    IntegratedDataset_test.to_csv("../data/dataset/test/IntegratedDataset.csv",index=False)

def generateLatestDataset():
    # Load DWD Wind Data
    dwd_Hornsea1 = xr.open_dataset("../data/raw/dwd_icon_eu_hornsea_1_20240129_20240519.nc")
    dwd_Hornsea1_features=preProcessNWPData(dwd_Hornsea1,featureType="wind",featureName=["WindSpeed:100"])

    # Load DWD Solar Data
    dwd_solar=xr.open_dataset("../data/raw/dwd_icon_eu_pes10_20240129_20240519.nc")
    dwd_solar_features=preProcessNWPData(dwd_solar,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    #Load GFS Wind Data
    gfs_Hornsea1 = xr.open_dataset("../data/raw/ncep_gfs_hornsea_1_20240129_20240519.nc")
    gfs_Hornsea1_features=preProcessNWPData(gfs_Hornsea1,featureType="wind",featureName=["WindSpeed:100"])

    #Load GFS Solar Data
    gfs_solar=xr.open_dataset("../data/raw/ncep_gfs_pes10_20240129_20240519.nc")
    gfs_solar_features=preProcessNWPData(gfs_solar,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    # Load Energy Data
    energy_data = pd.read_csv("../data/raw/Energy_Data_20240119_20240519.csv")
    energy_data = preProcessEnergyData(energy_data)

    # Generate Integrated Dataset
    IntegratedDataset_dwd=generateIntegratedDataset(dwd_Hornsea1_features,dwd_solar_features,energy_data,"dwd")
    IntegratedDataset_gfs=generateIntegratedDataset(gfs_Hornsea1_features,gfs_solar_features,energy_data,"gfs")


    # Generate Testset
    IntegratedDataset_dwd=generateTestset(IntegratedDataset_dwd,start_time="2024-02-19 23:00",end_time="2024-05-19 22:30").reset_index(drop=True)
    IntegratedDataset_gfs=generateTestset(IntegratedDataset_gfs,start_time="2024-02-19 23:00",end_time="2024-05-19 22:30").reset_index(drop=True)

    #Filling in missing values
    time_series=pd.date_range(start="2024-02-19 23:00",end="2024-05-19 22:30",freq="30T",tz="UTC")
    IntegratedDataset_dwd = pd.merge(pd.DataFrame({'valid_datetime': time_series}), IntegratedDataset_dwd, on='valid_datetime', how='left')
    df_supplem1=pd.read_csv("../data/2024-05-04_Supplem.csv")
    df_supplem2=pd.read_csv("../data/2024-05-13_Supplem.csv")
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
    WindDataset_dwd,SolarDataset_dwd=generateWindSolarDataset(IntegratedDataset,source="dwd")
    WindDataset_gfs,SolarDataset_gfs=generateWindSolarDataset(IntegratedDataset,source="gfs")

    # Save Dataset
    if not os.path.exists("../data/dataset/latest/dwd"):
        os.makedirs("../data/dataset/latest/dwd")
    if not os.path.exists("../data/dataset/latest/gfs"):
        os.makedirs("../data/dataset/latest/gfs")
        
    IntegratedDataset.to_csv('../data/dataset/latest/IntegratedDataset.csv',index=False)
    WindDataset_dwd.to_csv('../data/dataset/latest/dwd/WindDataset.csv',index=False)
    SolarDataset_dwd.to_csv('../data/dataset/latest/dwd/SolarDataset.csv',index=False)
    WindDataset_gfs.to_csv('../data/dataset/latest/gfs/WindDataset.csv',index=False)
    SolarDataset_gfs.to_csv('../data/dataset/latest/gfs/SolarDataset.csv',index=False)


if __name__ == "__main__":
    
    generateFullDataset()
    generateTrainTestDataset()
    generateLatestDataset()
    