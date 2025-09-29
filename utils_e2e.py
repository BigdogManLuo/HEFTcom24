import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import holidays
import utils
import utils_forecasting
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)

def preProcessIntegratedDataset(caseNumber):
    
    if caseNumber==1:
        #Merging DWD and GFS datasets
        IntegratedDataset_dwd=pd.read_csv("../data/dataset/full/dwd/IntegratedDataset.csv")
        IntegratedDataset_gfs=pd.read_csv("../data/dataset/full/gfs/IntegratedDataset.csv")
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
        columns_wind_dwd_features=pd.read_csv("../data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
        columns_wind_gfs_features=pd.read_csv("../data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
        columns_solar_dwd_features=pd.read_csv("../data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

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
        IntegratedDataset["power_pred"]=utils_forecasting.forecast_bidding(**params).values

        # Optimal Revenue
        optimalBidding=IntegratedDataset["total_generation_MWh"]+7.14*IntegratedDataset["Price_diff"]
        optimalBidding[optimalBidding<0]=0
        optimalBidding[optimalBidding>1800]=1800
        IntegratedDataset["OptimalRevenue"]=utils.getRevenue(Xb=optimalBidding,Xa=IntegratedDataset["total_generation_MWh"],yd=IntegratedDataset["DA_Price"],ys=IntegratedDataset["SS_Price"])
        
    elif caseNumber==2:
        
        IntegratedDataset_latest=pd.read_csv("../data/dataset/latest/IntegratedDataset.csv")
        IntegratedDataset_latest["Price_diff"]=IntegratedDataset_latest["DA_Price"]-IntegratedDataset_latest["SS_Price"]

        #Power Forecasting
        columns_wind_dwd_features=pd.read_csv("../data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
        columns_wind_gfs_features=pd.read_csv("../data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
        columns_solar_dwd_features=pd.read_csv("../data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

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
        IntegratedDataset_latest["power_pred"]=utils_forecasting.forecast_bidding(**params).values


        # History IntegratedDataset

        IntegratedDataset_dwd=pd.read_csv("../data/dataset/full/dwd/IntegratedDataset.csv")
        IntegratedDataset_gfs=pd.read_csv("../data/dataset/full/gfs/IntegratedDataset.csv")
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
        columns_wind_dwd_features=pd.read_csv("../data/dataset/full/dwd/WindDataset.csv").columns.tolist()[:-1]
        columns_wind_gfs_features=pd.read_csv("../data/dataset/full/gfs/WindDataset.csv").columns.tolist()[:-1]
        columns_solar_dwd_features=pd.read_csv("../data/dataset/full/dwd/SolarDataset.csv").columns.tolist()[:-1]

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
        IntegratedDataset["power_pred"]=utils_forecasting.forecast_bidding(**params).values


        # Merge IntegratedDataset
        IntegratedDataset=pd.concat([IntegratedDataset,IntegratedDataset_latest],axis=0).reset_index(drop=True)
        IntegratedDataset.dropna(axis=1,how="any",inplace=True)
        
        
        # Optimal Revenue
        optimalBidding=IntegratedDataset["total_generation_MWh"]+7.14*IntegratedDataset["Price_diff"]
        optimalBidding[optimalBidding<0]=0
        optimalBidding[optimalBidding>1800]=1800
        IntegratedDataset["OptimalRevenue"]=utils.getRevenue(Xb=optimalBidding,Xa=IntegratedDataset["total_generation_MWh"],yd=IntegratedDataset["DA_Price"],ys=IntegratedDataset["SS_Price"])
        
    else:
        raise ValueError("caseNumber must be 1 or 2")
    
    return IntegratedDataset

def generateModellingTable(caseNumber):
    
    IntegratedDataset=preProcessIntegratedDataset(caseNumber)
    
    #Modelling table
    modelling_table=IntegratedDataset[["valid_datetime","hours","power_pred","Price_diff","total_generation_MWh","DA_Price","SS_Price","OptimalRevenue"]].reset_index(drop=True)

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

def trainTestSplit(modelling_table,caseNumber,is_E2E):
    
    if caseNumber==1: #History
        IntegratedDataset_test=pd.read_csv("../data/dataset/test/IntegratedDataset.csv")
        idxs_test=modelling_table["valid_datetime"].isin(IntegratedDataset_test["valid_datetime"])
        test_data=modelling_table[idxs_test].copy()
        train_data=modelling_table[~idxs_test].copy()
    elif caseNumber==2: #Latest
        test_data=modelling_table.iloc[57432:,:]
        train_data=modelling_table.iloc[40000:57432,:]
        try:
            modelling_table.drop("valid_datetime",axis=1,inplace=True)
        except:
            pass
    else:
        raise ValueError("caseNumber must be 1 or 2")
    
    if is_E2E:
        columns_labels="OptimalRevenue" #Predict the bidding
    else:
        columns_labels="Price_diff" #Predict the price difference
        
    columns_features_train=train_data.columns.tolist()
    columns_features_test=train_data.columns.tolist()
    columns_features_train.remove("Price_diff")
    columns_features_train.remove("DA_Price")
    columns_features_train.remove("SS_Price")
    columns_features_train.remove("OptimalRevenue")
    columns_features_train.remove("power_error")
    columns_features_train.remove("power_pred")
    columns_features_test.remove("Price_diff")
    columns_features_test.remove("DA_Price")
    columns_features_test.remove("SS_Price")
    columns_features_test.remove("OptimalRevenue")
    columns_features_test.remove("power_error")
    columns_features_test.remove("total_generation_MWh")
    try:
        columns_features_train.remove("valid_datetime")
        columns_features_test.remove("valid_datetime")
    except:
        pass
    
    features_train=train_data[columns_features_train].values
    features_test=test_data[columns_features_test].values
    labels_train=train_data[columns_labels].values
    labels_test=test_data[columns_labels].values
    
    #Normalization
    scaler_features=StandardScaler()
    features_train[:,0]=scaler_features.fit_transform(features_train[:,0].reshape(-1,1)).squeeze()
    features_test[:,0]=scaler_features.transform(features_test[:,0].reshape(-1,1)).squeeze()
    
    scaler_labels=StandardScaler()
    if not is_E2E:
        labels_train=scaler_labels.fit_transform(labels_train.reshape(-1,1)).squeeze()
        labels_test=scaler_labels.transform(labels_test.reshape(-1,1)).squeeze()
        
    return features_train,features_test,labels_train,labels_test,train_data,test_data, scaler_labels
    

#Trading Loss
def trading_loss(pred_biddding,actual_generation,da_price,ss_price,optimal_revenue):
    
    #yd*Xb+(Xa-Xb)*ys-0.07*(Xa-Xb)**2
    pred_revenue=da_price*pred_biddding+(actual_generation-pred_biddding)*ss_price-0.07*(actual_generation-pred_biddding)**2
    loss = torch.mean(torch.abs(pred_revenue - optimal_revenue))
    return loss

class NNDataset(Dataset):

    def __init__(self, features, labels, actual_generation, da_price, ss_price):
        self.features = np.array(features, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.float32)
        self.actual_generation = np.array(actual_generation, dtype=np.float32)
        self.da_price = np.array(da_price, dtype=np.float32)
        self.ss_price = np.array(ss_price, dtype=np.float32)
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32), torch.tensor(self.actual_generation[idx], dtype=torch.float32,requires_grad=False), torch.tensor(self.da_price[idx], dtype=torch.float32,requires_grad=False), torch.tensor(self.ss_price[idx], dtype=torch.float32,requires_grad=False)

def try_resume(model, path):
    try:
        model.load_state_dict(torch.load(path))
        return True
    except Exception:
        return False

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

def train_model(features_train, labels_train, train_data, model_params, train_params, save_path, is_E2E=False):
    
    model = MLPModel(**model_params).to(train_params['device'])
    status = try_resume(model, save_path)
    if not status:
        train_dataset = NNDataset(
            features=features_train,
            labels=labels_train,
            actual_generation=train_data["total_generation_MWh"].values,
            da_price=train_data["DA_Price"].values,
            ss_price=train_data["SS_Price"].values
        )
        train_dataloader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True, drop_last=True)
        model = MLPModel(**model_params).to(train_params['device'])
        optimizer = optim.Adam(model.parameters(), lr=train_params['lr'])
        loss_func = nn.MSELoss() if not is_E2E else trading_loss
        for epoch in tqdm(range(train_params['num_epochs'])):
            model.train()
            loss_train = 0
            for inputs, targets, actual_generation, da_price, ss_price in train_dataloader:
                inputs, targets = inputs.to(train_params['device']), targets.to(train_params['device'])
                actual_generation = actual_generation.to(train_params['device'])
                da_price = da_price.to(train_params['device'])
                ss_price = ss_price.to(train_params['device'])
                optimizer.zero_grad()
                outputs = model(inputs)
                if is_E2E:
                    loss = loss_func(outputs.squeeze(), actual_generation, da_price, ss_price, targets)
                else:
                    loss = loss_func(outputs.squeeze(), targets)
                loss.backward()
                optimizer.step()
                loss_train += loss.item()
            loss_train /= len(train_dataloader)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    
    return model
    
def test_model(model, features_test, scaler_labels, test_data, device, is_E2E=False):
    model.eval()
    with torch.no_grad():
        outputs = model(torch.tensor(np.array(features_test, dtype=np.float32), dtype=torch.float32, device=device))
        if not is_E2E:
            outputs = scaler_labels.inverse_transform(outputs.squeeze().cpu().numpy().reshape(-1, 1)).reshape(-1)
            biddings = test_data["power_pred"] + 7.14 * outputs
            biddings = np.clip(biddings, 0, 1800)
            Revenue = utils.getRevenue(biddings, test_data["total_generation_MWh"], test_data["DA_Price"], test_data["SS_Price"])
            return Revenue, outputs
        else:
            outputs = outputs.squeeze().cpu().numpy().reshape(-1)
            outputs = np.clip(outputs, 0, 1800)
            pd_pred = utils.getEquivalentPriceSpreadForecast(Xb=outputs, Xa=test_data["power_pred"].values)
            Revenue = utils.getRevenue(outputs, test_data["total_generation_MWh"], test_data["DA_Price"], test_data["SS_Price"])
            return Revenue, pd_pred