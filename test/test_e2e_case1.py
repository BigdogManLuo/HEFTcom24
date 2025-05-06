import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from utils_e2e import *
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


torch.manual_seed(0)
device='cuda' if torch.cuda.is_available() else 'cpu'

def train():

    train_dataset = NNDataset(features=features_train, labels=labels_train,actual_generation=train_data["total_generation_MWh"].values, da_price=train_data["DA_Price"].values, ss_price=train_data["SS_Price"].values)
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=True,drop_last=True)

    model = MLPModel(input_size=features_train.shape[1], hidden_layer_size=32, output_size=1,num_layers=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    num_epochs = 40
    for epoch in range(num_epochs):
        model.train()
        loss_train=0
        for inputs, targets, actual_generation, da_price, ss_price in train_dataloader:
            inputs, targets, actual_generation, da_price, ss_price = inputs.to(device), targets.to(device), actual_generation.to(device), da_price.to(device), ss_price.to(device)
            optimizer.zero_grad()
            outputs = model(inputs) #forecasting bidddind
            loss = trading_loss(outputs.squeeze(), actual_generation, da_price, ss_price,targets)
            loss.backward()
            optimizer.step()
            loss_train+=loss.item()
        loss_train/=len(train_dataloader)

    # Save model
    if not os.path.exists("../models/Prices/history"):
        os.makedirs("../models/Prices/history")
    torch.save(model.state_dict(), "../models/Prices/history/e2e.pth")


# Preparing dataset
IntegratedDataset, modelling_table=generateModellingTable(caseNumber=1)
features_train,features_test,labels_train,labels_test,train_data,test_data,scaler_labels=trainTestSplit(modelling_table,caseNumber=1,is_E2E=True)


# Training
if not os.path.exists("../models/Prices/history/e2e.pth"):
    train()

# Testing
model = MLPModel(input_size=features_train.shape[1], hidden_layer_size=32, output_size=1,num_layers=2).to(device)
model.load_state_dict(torch.load("../models/Prices/history/e2e.pth"))
model.eval()
with torch.no_grad():
    outputs = model(torch.tensor(np.array(features_test,dtype=np.float32),dtype=torch.float32,device=device))
    #Inverse transform the outputs
    outputs=outputs.squeeze().cpu().numpy().reshape(-1)
    outputs[outputs<0]=0
    outputs[outputs>1800]=1800
    pd_pred=utils.getEquivalentPriceSpreadForecast(Xb=outputs,Xa=test_data["power_pred"].values)
    Revenue=utils.getRevenue(outputs,test_data["total_generation_MWh"],test_data["DA_Price"],test_data["SS_Price"])

print("Revenue: ", Revenue.sum())

if not os.path.exists("../data/revenues/case1"):
    os.makedirs("../data/revenues/case1")
np.save("../data/revenues/case1/Revenue_e2e.npy", Revenue)
np.save("../data/revenues/case1/Revenue_e2e.npy", Revenue)
np.save("../data/revenues/case1/pd_pred_e2e.npy", pd_pred)