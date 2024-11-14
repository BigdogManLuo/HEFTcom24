import xarray as xr
import os
from utils_dp import *
   
def generateDataset():
    # Load DWD Wind Data
    dwd_Hornsea1_old = xr.open_dataset("data/raw/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
    dwd_Hornsea1_new = xr.open_dataset("data/raw/dwd_icon_eu_hornsea_1_20231027_20240108.nc")
    dwd_Hornsea1_latest = xr.open_dataset("data/raw/dwd_icon_eu_hornsea_1_20240108_20240129.nc")

    dwd_Hornsea1_features_old=preProcessNWPData(dwd_Hornsea1_old,featureType="wind",featureName=["WindSpeed:100"])
    dwd_Hornsea1_features_new=preProcessNWPData(dwd_Hornsea1_new,featureType="wind",featureName=["WindSpeed:100"])
    dwd_Hornsea1_features_latest=preProcessNWPData(dwd_Hornsea1_latest,featureType="wind",featureName=["WindSpeed:100"])

    dwd_Hornsea1_features_old = dwd_Hornsea1_features_old[dwd_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    dwd_Hornsea1_features_new = dwd_Hornsea1_features_new[dwd_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    dwd_Hornsea1_features=pd.concat([dwd_Hornsea1_features_old,dwd_Hornsea1_features_new,dwd_Hornsea1_features_latest],axis=0).reset_index(drop=True)

    # Load DWD Solar Data
    dwd_solar_old=xr.open_dataset("data/raw/dwd_icon_eu_pes10_20200920_20231027.nc")
    dwd_solar_new=xr.open_dataset("data/raw/dwd_icon_eu_pes10_20231027_20240108.nc")
    dwd_solar_latest=xr.open_dataset("data/raw/dwd_icon_eu_pes10_20240108_20240129.nc")

    dwd_solar_features_old=preProcessNWPData(dwd_solar_old,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    dwd_solar_features_new=preProcessNWPData(dwd_solar_new,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    dwd_solar_features_latest=preProcessNWPData(dwd_solar_latest,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    dwd_solar_features_old = dwd_solar_features_old[dwd_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    dwd_solar_features_new = dwd_solar_features_new[dwd_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    dwd_solar_features=pd.concat([dwd_solar_features_old,dwd_solar_features_new,dwd_solar_features_latest],axis=0).reset_index(drop=True)


    # Load GFS Wind Data
    gfs_Hornsea1_old = xr.open_dataset("data/raw/ncep_gfs_hornsea_1_20200920_20231027.nc")
    gfs_Hornsea1_new = xr.open_dataset("data/raw/ncep_gfs_hornsea_1_20231027_20240108.nc")
    gfs_Hornsea1_latest = xr.open_dataset("data/raw/ncep_gfs_hornsea_1_20240108_20240129.nc")

    gfs_Hornsea1_features_old=preProcessNWPData(gfs_Hornsea1_old,featureType="wind",featureName=["WindSpeed:100"])
    gfs_Hornsea1_features_new=preProcessNWPData(gfs_Hornsea1_new,featureType="wind",featureName=["WindSpeed:100"])
    gfs_Hornsea1_features_latest=preProcessNWPData(gfs_Hornsea1_latest,featureType="wind",featureName=["WindSpeed:100"])

    gfs_Hornsea1_features_old = gfs_Hornsea1_features_old[gfs_Hornsea1_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    gfs_Hornsea1_features_new = gfs_Hornsea1_features_new[gfs_Hornsea1_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    gfs_Hornsea1_features=pd.concat([gfs_Hornsea1_features_old,gfs_Hornsea1_features_new,gfs_Hornsea1_features_latest],axis=0).reset_index(drop=True)

    # Load GFS Solar Data
    gfs_solar_old=xr.open_dataset("data/raw/ncep_gfs_pes10_20200920_20231027.nc")
    gfs_solar_new=xr.open_dataset("data/raw/ncep_gfs_pes10_20231027_20240108.nc")
    gfs_solar_latest=xr.open_dataset("data/raw/ncep_gfs_pes10_20240108_20240129.nc")

    gfs_solar_features_old=preProcessNWPData(gfs_solar_old,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    gfs_solar_features_new=preProcessNWPData(gfs_solar_new,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])
    gfs_solar_features_latest=preProcessNWPData(gfs_solar_latest,featureType="solar",featureName=["SolarDownwardRadiation","CloudCover"])

    gfs_solar_features_old = gfs_solar_features_old[gfs_solar_features_old["ref_datetime"] < "2023-10-27 00:00:00"]
    gfs_solar_features_new = gfs_solar_features_new[gfs_solar_features_new["ref_datetime"] < "2024-01-08 00:00:00"]
    gfs_solar_features=pd.concat([gfs_solar_features_old,gfs_solar_features_new,gfs_solar_features_latest],axis=0).reset_index(drop=True)

    # Load Energy Data
    energy_data = pd.read_csv("data/raw/Energy_Data_20200920_20240118.csv")
    energy_data = preProcessEnergyData(energy_data)

    # Generate Integrated Dataset
    IntegratedDataset_dwd=generateIntegratedDataset(dwd_Hornsea1_features,dwd_solar_features,energy_data,"dwd")
    IntegratedDataset_gfs=generateIntegratedDataset(gfs_Hornsea1_features,gfs_solar_features,energy_data,"gfs")

    #Generate Wind and Solar Dataset
    WindDataset_dwd,SolarDataset_dwd=generateWindSolarDataset(IntegratedDataset_dwd,"dwd")
    WindDataset_gfs,SolarDataset_gfs=generateWindSolarDataset(IntegratedDataset_gfs,"gfs")


    #Save Dataset
    if not os.path.exists("data/dataset/full/dwd"):
        os.makedirs("data/dataset/full/dwd")
    IntegratedDataset_dwd.to_csv('data/dataset/full/dwd/IntegratedDataset.csv',index=False)
    WindDataset_dwd.to_csv('data/dataset/full/dwd/WindDataset.csv',index=False)
    SolarDataset_dwd.to_csv('data/dataset/full/dwd/SolarDataset.csv',index=False)

    if not os.path.exists("data/dataset/full/gfs"):
        os.makedirs("data/dataset/full/gfs")
    IntegratedDataset_gfs.to_csv('data/dataset/full/gfs/IntegratedDataset.csv',index=False)
    WindDataset_gfs.to_csv('data/dataset/full/gfs/WindDataset.csv',index=False)
    SolarDataset_gfs.to_csv('data/dataset/full/gfs/SolarDataset.csv',index=False)

if __name__ == "__main__":
    generateDataset()
