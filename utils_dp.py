import pandas as pd
from tqdm import tqdm
import numpy as np

def extractFeaturesFromNWPData(NWPNCData,featureType,featureName):
    
    if featureType=="wind":
        dim=["latitude","longitude"]
    elif featureType=="solar":
        dim="point"

    #spatial feature 
    meanFeatures=NWPNCData[featureName].mean(dim=dim).to_dataframe().reset_index()
    maxFeatures=NWPNCData[featureName].max(dim=dim).to_dataframe().reset_index().rename(columns={feature_name:"max"+feature_name for feature_name in featureName})
    minFeatures=NWPNCData[featureName].min(dim=dim).to_dataframe().reset_index().rename(columns={feature_name:"min"+feature_name for feature_name in featureName})
    
    #Rename
    if "ref_datetime" not in meanFeatures.columns and "valid_datetime" not in meanFeatures.columns:
        meanFeatures.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
        maxFeatures.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)
        minFeatures.rename(columns={"reference_time":"ref_datetime","valid_time":"valid_datetime"},inplace=True)

    #Merge Features
    NWPData=meanFeatures.merge(maxFeatures,how="left",on=["ref_datetime","valid_datetime"])
    NWPData=NWPData.merge(minFeatures,how="left",on=["ref_datetime","valid_datetime"])

    return NWPData

def preProcessNWPData(NWPNCData,featureType,featureName):
    
    NWPData=extractFeaturesFromNWPData(NWPNCData,featureType,featureName)

    #Standardize time
    NWPData["ref_datetime"] = NWPData["ref_datetime"].dt.tz_localize("UTC")
    NWPData["valid_datetime"] = NWPData["ref_datetime"] + pd.TimedeltaIndex(NWPData["valid_datetime"],unit="hours")

    #lead time <= 48h
    NWPData=NWPData[NWPData["valid_datetime"] - NWPData["ref_datetime"] < np.timedelta64(48,"h")].reset_index(drop=True)

    #Interpolate
    NWPData=NWPData.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
    NWPData = NWPData.drop(columns="ref_datetime",axis=1).reset_index()

    return NWPData

def preProcessEnergyData(energy_data):
    energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
    energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
    energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]
    energy_data["total_generation_MWh"]=energy_data["Wind_MWh_credit"]+energy_data["Solar_MWh_credit"]
    energy_data = energy_data[["dtm","Wind_MWh_credit","Solar_MWh_credit","total_generation_MWh","DA_Price","SS_Price"]]

    return energy_data

def generateIntegratedDataset(dwd_Hornsea1_features,dwd_solar_features,energy_data,source):

    # Merge Data
    modelling_table=dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
    modelling_table= modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")
    modelling_table.drop(columns=["dtm"],inplace=True)
    # Generate Integrated Dataset
    IntegratedDataset=pd.DataFrame()
    modelling_table=modelling_table.groupby("ref_datetime")
    print("Generating Integrated Dataset......")
    for group_idx,(ref_datetime,group) in enumerate(tqdm(modelling_table)):
        
        group[f"ws_100_t-1_{source}_1"] = group["WindSpeed:100"].shift(1)
        group[f"ws_100_t-1_{source}_max"] = group["maxWindSpeed:100"].shift(1)
        group[f"ws_100_t-1_{source}_min"] = group["minWindSpeed:100"].shift(1)
        group[f"ws_100_t_{source}_1"] = group["WindSpeed:100"]
        group[f"ws_100_t_{source}_max"] = group["maxWindSpeed:100"]
        group[f"ws_100_t_{source}_min"] = group["minWindSpeed:100"]
        group[f"ws_100_t+1_{source}_1"] = group["WindSpeed:100"].shift(-1)
        group[f"ws_100_t+1_{source}_max"] = group["maxWindSpeed:100"].shift(-1)
        group[f"ws_100_t+1_{source}_min"] = group["minWindSpeed:100"].shift(-1)
        group[f"rad_t-1_{source}"] = group["SolarDownwardRadiation"].shift(1)
        group[f"rad_t-1_{source}_max"] = group["maxSolarDownwardRadiation"].shift(1)
        group[f"rad_t-1_{source}_min"] = group["minSolarDownwardRadiation"].shift(1)
        group[f"rad_t_{source}"] = group["SolarDownwardRadiation"]
        group[f"rad_t_{source}_max"] = group["maxSolarDownwardRadiation"]
        group[f"rad_t_{source}_min"] = group["minSolarDownwardRadiation"]
        group[f"rad_t+1_{source}"] = group["SolarDownwardRadiation"].shift(-1)
        group[f"rad_t+1_{source}_max"] = group["maxSolarDownwardRadiation"].shift(-1)
        group[f"rad_t+1_{source}_min"] = group["minSolarDownwardRadiation"].shift(-1)
        group[f"cloudcov_t-1_{source}"] = group["CloudCover"].shift(1)
        group[f"cloudcov_t-1_{source}_max"] = group["maxCloudCover"].shift(1)
        group[f"cloudcov_t-1_{source}_min"] = group["minCloudCover"].shift(1)
        group[f"cloudcov_t_{source}"] = group["CloudCover"]
        group[f"cloudcov_t_{source}_max"] = group["maxCloudCover"]
        group[f"cloudcov_t_{source}_min"] = group["minCloudCover"]
        group[f"cloudcov_t+1_{source}"] = group["CloudCover"].shift(-1)
        group[f"cloudcov_t+1_{source}_max"] = group["maxCloudCover"].shift(-1)
        group[f"cloudcov_t+1_{source}_min"] = group["minCloudCover"].shift(-1)

        group.drop(columns=["WindSpeed:100","maxWindSpeed:100","minWindSpeed:100","SolarDownwardRadiation","maxSolarDownwardRadiation","minSolarDownwardRadiation","CloudCover","maxCloudCover","minCloudCover"],inplace=True)

        IntegratedDataset=pd.concat([IntegratedDataset,group],axis=0)

    IntegratedDataset=IntegratedDataset.reset_index(drop=True)
    IntegratedDataset=IntegratedDataset.dropna(axis=0,how='any')
    IntegratedDataset=IntegratedDataset[IntegratedDataset["Wind_MWh_credit"]<=620]
    IntegratedDataset["hours"]=pd.to_datetime(IntegratedDataset["valid_datetime"]).dt.hour

    return IntegratedDataset

def generateWindSolarDataset(IntegratedDataset,source):
    columns_wind_features=[f"ws_100_t-1_{source}_1",f"ws_100_t-1_{source}_max",f"ws_100_t-1_{source}_min",
                       f"ws_100_t_{source}_1",f"ws_100_t_{source}_max",f"ws_100_t_{source}_min",
                       f"ws_100_t+1_{source}_1",f"ws_100_t+1_{source}_max",f"ws_100_t+1_{source}_min"]
    columns_wind_labels=["Wind_MWh_credit"]
    columns_solar_features=[f"rad_t-1_{source}",f"rad_t-1_{source}_max",f"rad_t-1_{source}_min",f"cloudcov_t-1_{source}",f"cloudcov_t-1_{source}_max",f"cloudcov_t-1_{source}_min",
                        f"rad_t_{source}",f"rad_t_{source}_max",f"rad_t_{source}_min",f"cloudcov_t_{source}",f"cloudcov_t_{source}_max",f"cloudcov_t_{source}_min",
                        "hours"]
    columns_solar_labels=["Solar_MWh_credit"]   

    WindDataset=IntegratedDataset[columns_wind_features+columns_wind_labels]
    SolarDataset=IntegratedDataset[columns_solar_features+columns_solar_labels]

    return WindDataset,SolarDataset

def generateTestset(IntegratedDataset,start_time,end_time):

    IntegratedDataset["ref_datetime"] = pd.to_datetime(IntegratedDataset["ref_datetime"])
    IntegratedDataset["valid_datetime"] = pd.to_datetime(IntegratedDataset["valid_datetime"])
    IntegratedDataset=IntegratedDataset[IntegratedDataset["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)

    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])<=np.timedelta64(47,"h")]
    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])>=np.timedelta64(23,"h")]

    IntegratedDataset_test = IntegratedDataset[(IntegratedDataset["valid_datetime"]>=start_time) & (IntegratedDataset["valid_datetime"]<=end_time)]

    return IntegratedDataset_test

def generateTrainset(IntegratedDataset_full,IntegratedDataset_test):

    IntegratedDataset_full["ref_datetime"] = pd.to_datetime(IntegratedDataset_full["ref_datetime"])
    IntegratedDataset_full["valid_datetime"] = pd.to_datetime(IntegratedDataset_full["valid_datetime"])

    IntegratedDataset_train = IntegratedDataset_full[~IntegratedDataset_full["valid_datetime"].isin(IntegratedDataset_test["valid_datetime"])]

    return IntegratedDataset_train
