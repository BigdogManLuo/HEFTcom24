import pandas as pd
from comp_utils import getLatestSolarGeneration,getSolarDatasetfromNC


def mergeToDataset(modelling_table,target_type):

    Dataset_full=pd.read_csv(f"data/dataset/full/gfs/{target_type.capitalize()}Dataset.csv")
    Dataset_full=pd.concat([Dataset_full,modelling_table],axis=0)
    Dataset_full.to_csv(f"data/dataset/full/gfs/{target_type.capitalize()}Dataset.csv",index=False)
    
    Dataset_train=pd.read_csv(f"data/dataset/train/gfs/{target_type.capitalize()}Dataset.csv")
    Dataset_train=pd.concat([Dataset_train,modelling_table],axis=0)
    Dataset_train.to_csv(f"data/dataset/train/gfs/{target_type.capitalize()}Dataset.csv",index=False)
    
    return Dataset_full,Dataset_train


if __name__ == "__main__":
    SolarGeneration=getLatestSolarGeneration()
    modelling_table_solar=getSolarDatasetfromNC(SolarGeneration)
    SolarDataset_full,SolarDataset_train=mergeToDataset(modelling_table_solar,"solar")
    
    
    
    
    