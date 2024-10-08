import pandas as pd
import numpy as np
import os
from utils_dp import generateWindSolarDataset,generateTestset,generateTrainset

if __name__ == "__main__":

    IntegratedDataset_dwd = pd.read_csv("data/dataset/full/dwd/IntegratedDataset.csv")    
    IntegratedDataset_gfs = pd.read_csv("data/dataset/full/gfs/IntegratedDataset.csv")    

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
    if not os.path.exists("data/dataset/train/dwd"):
        os.makedirs("data/dataset/train/dwd")
    if not os.path.exists("data/dataset/train/gfs"):
        os.makedirs("data/dataset/train/gfs")
    if not os.path.exists("data/dataset/test/dwd"):
        os.makedirs("data/dataset/test/dwd")
    if not os.path.exists("data/dataset/test/gfs"):
        os.makedirs("data/dataset/test/gfs")

    WindDataset_train_dwd.to_csv("data/dataset/train/dwd/WindDataset.csv",index=False)
    SolarDataset_train_dwd.to_csv("data/dataset/train/dwd/SolarDataset.csv",index=False)
    WindDataset_train_gfs.to_csv("data/dataset/train/gfs/WindDataset.csv",index=False)
    SolarDataset_train_gfs.to_csv("data/dataset/train/gfs/SolarDataset.csv",index=False)
    WindDataset_test_dwd.to_csv("data/dataset/test/dwd/WindDataset.csv",index=False)
    SolarDataset_test_dwd.to_csv("data/dataset/test/dwd/SolarDataset.csv",index=False)
    WindDataset_test_gfs.to_csv("data/dataset/test/gfs/WindDataset.csv",index=False)
    SolarDataset_test_gfs.to_csv("data/dataset/test/gfs/SolarDataset.csv",index=False)
    IntegratedDataset_test.to_csv("data/dataset/test/IntegratedDataset.csv",index=False)


    
    