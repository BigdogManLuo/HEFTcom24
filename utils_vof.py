import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import holidays
import utils
from sklearn.preprocessing import StandardScaler,MinMaxScaler

def generateModellingTableHistory():

    #Merging DWD and GFS datasets
    IntegratedDataset_dwd=pd.read_csv("data/dataset/full/dwd/IntegratedDataset.csv")
    IntegratedDataset_gfs=pd.read_csv("data/dataset/full/gfs/IntegratedDataset.csv")
    IntegratedDataset=IntegratedDataset_gfs.merge(IntegratedDataset_dwd,how="inner",on=["ref_datetime","valid_datetime"])
    IntegratedDataset.rename(columns={"Wind_MWh_credit_x":"Wind_MWh_credit","Solar_MWh_credit_x":"Solar_MWh_credit","total_generation_MWh_x":"total_generation_MWh","hours_x":"hours","DA_Price_x":"DA_Price","SS_Price_x":"SS_Price","dtm_x":"dtm"},inplace=True)

    #Test Environment
    IntegratedDataset["ref_datetime"] = pd.to_datetime(IntegratedDataset["ref_datetime"])
    IntegratedDataset["valid_datetime"] = pd.to_datetime(IntegratedDataset["valid_datetime"])
    IntegratedDataset=IntegratedDataset[IntegratedDataset["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)
    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])<=np.timedelta64(47,"h")]
    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])>=np.timedelta64(23,"h")]

    IntegratedDataset["Price_diff"]=IntegratedDataset["DA_Price"]-IntegratedDataset["SS_Price"]

    #Power Forecasting
    columns_wind_dwd_features=pd.read_csv("data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
    columns_wind_gfs_features=pd.read_csv("data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
    columns_solar_dwd_features=pd.read_csv("data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

    features_wind_dwd=IntegratedDataset[columns_wind_dwd_features].values
    features_wind_gfs=IntegratedDataset[columns_wind_gfs_features].values
    features_solar_dwd=IntegratedDataset[columns_solar_dwd_features].values
    params={
        "wind_features_dwd":features_wind_dwd,
        "wind_features_gfs":features_wind_gfs,
        "solar_features":features_solar_dwd,
        "full":False,
        "hours":IntegratedDataset["hours"].values,
        "WLimit":False,
        "SolarRevise":False
    }
    IntegratedDataset["power_pred"]=utils.forecast_bidding(**params).values

    #Modelling table
    modelling_table=IntegratedDataset[["valid_datetime","hours","power_pred","Price_diff","total_generation_MWh"]].reset_index(drop=True)

    #Feature Engineering
    modelling_table["valid_datetime"]=pd.to_datetime(modelling_table["valid_datetime"])
    modelling_table["season"]=modelling_table["valid_datetime"].dt.quarter.astype(int)
    #modelling_table["is_weekend"]=modelling_table["valid_datetime"].dt.weekday>=5
    #modelling_table["is_weekend"]=modelling_table["is_weekend"].astype(int)
    uk_holidays = holidays.UnitedKingdom()
    #modelling_table["is_holiday"]=modelling_table["valid_datetime"].dt.date.apply(lambda x: x in uk_holidays).astype(int)
    modelling_table=pd.concat([modelling_table,pd.get_dummies(modelling_table["hours"],prefix="hours").astype(int)],axis=1)
    modelling_table=pd.concat([modelling_table,pd.get_dummies(modelling_table["season"],prefix="season").astype(int)],axis=1)
    modelling_table["power_error"]=modelling_table["total_generation_MWh"]-modelling_table["power_pred"]
    modelling_table.drop("hours",axis=1,inplace=True)
    modelling_table.drop("season",axis=1,inplace=True)
    #modelling_table.drop("Total_MWh",axis=1,inplace=True)
    modelling_table=modelling_table.dropna()

    return IntegratedDataset, modelling_table

def generateModellingTableLatest():
    
    IntegratedDataset_latest=pd.read_csv("data/dataset/latest/IntegratedDataset.csv")
    IntegratedDataset_latest["Price_diff"]=IntegratedDataset_latest["DA_Price"]-IntegratedDataset_latest["SS_Price"]

    #Power Forecasting
    columns_wind_dwd_features=pd.read_csv("data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
    columns_wind_gfs_features=pd.read_csv("data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
    columns_solar_dwd_features=pd.read_csv("data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

    features_wind_dwd=IntegratedDataset_latest[columns_wind_dwd_features].values
    features_wind_gfs=IntegratedDataset_latest[columns_wind_gfs_features].values
    features_solar_dwd=IntegratedDataset_latest[columns_solar_dwd_features].values

    params={
        "wind_features_dwd":features_wind_dwd,
        "wind_features_gfs":features_wind_gfs,
        "solar_features":features_solar_dwd,
        "full":True,
        "hours":IntegratedDataset_latest["hours"].values,
        "WLimit":True,
        "availableCapacity":IntegratedDataset_latest["availableCapacity"].values,
        "SolarRevise":True,
        "rolling_test":True,
    }
    IntegratedDataset_latest["power_pred"]=utils.forecast_bidding(**params).values


    #%% History IntegratedDataset

    IntegratedDataset_dwd=pd.read_csv("data/dataset/full/dwd/IntegratedDataset.csv")
    IntegratedDataset_gfs=pd.read_csv("data/dataset/full/gfs/IntegratedDataset.csv")
    IntegratedDataset=IntegratedDataset_gfs.merge(IntegratedDataset_dwd,how="inner",on=["ref_datetime","valid_datetime"])
    IntegratedDataset.rename(columns={"Wind_MWh_credit_x":"Wind_MWh_credit","Solar_MWh_credit_x":"Solar_MWh_credit","total_generation_MWh_x":"total_generation_MWh","hours_x":"hours","DA_Price_x":"DA_Price","SS_Price_x":"SS_Price","dtm_x":"dtm"},inplace=True)


    #Test Environment
    IntegratedDataset["ref_datetime"] = pd.to_datetime(IntegratedDataset["ref_datetime"])
    IntegratedDataset["valid_datetime"] = pd.to_datetime(IntegratedDataset["valid_datetime"])
    IntegratedDataset=IntegratedDataset[IntegratedDataset["ref_datetime"].dt.strftime("%H:%M")=="00:00"].reset_index(drop=True)
    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])<=np.timedelta64(47,"h")]
    IntegratedDataset = IntegratedDataset[(IntegratedDataset["valid_datetime"] - IntegratedDataset["ref_datetime"])>=np.timedelta64(23,"h")]

    IntegratedDataset["Price_diff"]=IntegratedDataset["DA_Price"]-IntegratedDataset["SS_Price"]

    #Power Forecasting
    columns_wind_dwd_features=pd.read_csv("data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
    columns_wind_gfs_features=pd.read_csv("data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
    columns_solar_dwd_features=pd.read_csv("data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

    features_wind_dwd=IntegratedDataset[columns_wind_dwd_features].values
    features_wind_gfs=IntegratedDataset[columns_wind_gfs_features].values
    features_solar_dwd=IntegratedDataset[columns_solar_dwd_features].values
    params={
        "wind_features_dwd":features_wind_dwd,
        "wind_features_gfs":features_wind_gfs,
        "solar_features":features_solar_dwd,
        "full":False,
        "hours":IntegratedDataset["hours"].values,
        "WLimit":False,
        "SolarRevise":False
    }
    IntegratedDataset["power_pred"]=utils.forecast_bidding(**params).values


    #%% Merge IntegratedDataset
    IntegratedDataset=pd.concat([IntegratedDataset,IntegratedDataset_latest],axis=0).reset_index(drop=True)
    IntegratedDataset.dropna(axis=1,how="any",inplace=True)

    #%% #Modelling table
    modelling_table=IntegratedDataset[["valid_datetime","hours","power_pred","Price_diff","total_generation_MWh"]].reset_index(drop=True)

    #Feature Engineering
    modelling_table["valid_datetime"]=pd.to_datetime(modelling_table["valid_datetime"])
    modelling_table["season"]=modelling_table["valid_datetime"].dt.quarter.astype(int)
    modelling_table=pd.concat([modelling_table,pd.get_dummies(modelling_table["hours"],prefix="hours").astype(int)],axis=1)
    modelling_table=pd.concat([modelling_table,pd.get_dummies(modelling_table["season"],prefix="season").astype(int)],axis=1)
    modelling_table.drop("hours",axis=1,inplace=True)
    modelling_table.drop("season",axis=1,inplace=True)
    modelling_table["power_error"]=modelling_table["total_generation_MWh"]-modelling_table["power_pred"]
    modelling_table=modelling_table.dropna()

    return IntegratedDataset, modelling_table


def trainTestSplitHistory(modelling_table):

    #Train-test split
    IntegratedDataset_test=pd.read_csv("data/dataset/test/IntegratedDataset.csv")
    idxs_test=modelling_table["valid_datetime"].isin(IntegratedDataset_test["valid_datetime"])
    test_data=modelling_table[idxs_test].copy()
    train_data=modelling_table[~idxs_test].copy()
    columns_labels="Price_diff"
    columns_features_train=train_data.columns.tolist()
    columns_features_test=train_data.columns.tolist()
    columns_features_train.remove("Price_diff")
    columns_features_train.remove("power_error")
    columns_features_train.remove("valid_datetime")
    columns_features_train.remove("power_pred")
    columns_features_test.remove("Price_diff")
    columns_features_test.remove("power_error")
    columns_features_test.remove("valid_datetime")
    columns_features_test.remove("total_generation_MWh")

    features_train=train_data[columns_features_train].values
    features_test=test_data[columns_features_test].values
    labels_train=train_data[columns_labels].values
    labels_test=test_data[columns_labels].values

    #Normalization
    scaler_features=StandardScaler()
    features_train[:,0]=scaler_features.fit_transform(features_train[:,0].reshape(-1,1)).squeeze()
    features_test[:,0]=scaler_features.transform(features_test[:,0].reshape(-1,1)).squeeze()

    return features_train,features_test,labels_train,labels_test,train_data,test_data

def trainTestSplitLatest(modelling_table):

    test_data=modelling_table.iloc[60312:,:]
    train_data=modelling_table.iloc[57432:60312,:]
    modelling_table.drop("valid_datetime",axis=1,inplace=True)

    columns_labels="Price_diff"
    columns_features_train=train_data.columns.tolist()
    columns_features_test=train_data.columns.tolist()
    columns_features_train.remove("Price_diff")
    columns_features_train.remove("power_error")
    columns_features_train.remove("valid_datetime")
    columns_features_train.remove("power_pred")
    columns_features_train.remove("total_generation_MWh")

    columns_features_test.remove("Price_diff")
    columns_features_test.remove("power_error")
    columns_features_test.remove("valid_datetime")
    columns_features_test.remove("total_generation_MWh")
    columns_features_test.remove("power_pred")

    features_train=train_data[columns_features_train].values
    features_test=test_data[columns_features_test].values
    labels_train=train_data[columns_labels].values
    labels_test=test_data[columns_labels].values

    #normalization
    scaler_features=StandardScaler()
    features_train[:,0]=scaler_features.fit_transform(features_train[:,0].reshape(-1,1)).squeeze()
    features_test[:,0]=scaler_features.transform(features_test[:,0].reshape(-1,1)).squeeze()

    return features_train,features_test,labels_train,labels_test,train_data,test_data

def generateTestPricesHistory(IntegratedDataset):
    
    IntegratedDataset_test=pd.read_csv("data/dataset/test/IntegratedDataset.csv")
    idxs_test=IntegratedDataset["valid_datetime"].isin(IntegratedDataset_test["valid_datetime"])
    DA_Price=IntegratedDataset.loc[idxs_test,"DA_Price"].values
    SS_Price=IntegratedDataset.loc[idxs_test,"SS_Price"].values
    return DA_Price,SS_Price

def generateTestPricesLatest(IntegratedDataset):

    DA_Price=IntegratedDataset.loc[60312:,"DA_Price"].values
    SS_Price=IntegratedDataset.loc[60312:,"SS_Price"].values
    return DA_Price,SS_Price


#Trading Loss
def trading_loss(power_error,pd_pred,Price_diff):
    #power_pred: power forecast
    #pd_pred: price difference forecast
    #total_generation_MWh: actual power generation
    #Price_diff: actual price difference
    loss=3.57*((Price_diff-pd_pred)**2)+power_error*(Price_diff-pd_pred)
    
    return loss.mean()

class NNDataset(Dataset):

    def __init__(self, features, labels, powererrs):
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        self.powererrs = np.array(powererrs, dtype=np.float32)
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32), torch.tensor(self.powererrs[idx], dtype=torch.float32)
    

#Multilayer Perceptron with batch normalization
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size,num_layers=3):
        super(MLPModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.input_layer = nn.Linear(input_size, hidden_layer_size)
        self.hidden_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU()
        ) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x


def preTrain(num_epochs,train_dataloader,device,optimizer,model,loss_func):

    for epoch in range(num_epochs):
        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        model.eval()

    return model
        
def train(num_epochs,train_dataloader,device,optimizer,model):

    for epoch in range(num_epochs):
        for inputs, targets, power_errors in train_dataloader:
            inputs, targets, power_errors = inputs.to(device), targets.to(device), power_errors.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = trading_loss(power_errors, outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

    return model

def testRevenue(model,features_test,test_data,DA_Price,SS_Price):
    model.eval()
    pd_pred=model(torch.tensor(np.array(features_test,dtype=np.float32),dtype=torch.float32,device=device)).detach().squeeze().cpu().numpy()
    biddings= test_data["power_pred"] + pd_pred
    biddings[biddings<0]=0
    biddings[biddings>1800]=1800
    Revenue=utils.getRevenue(biddings,test_data["total_generation_MWh"],DA_Price,SS_Price)
    return Revenue