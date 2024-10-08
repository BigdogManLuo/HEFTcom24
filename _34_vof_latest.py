import pandas as pd
import utils
from utils_vof import *
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#fix random seed
torch.manual_seed(0)

#%% Preparing dataset
IntegratedDataset, modelling_table=generateModellingTableLatest()
features_train,features_test,labels_train,labels_test,train_data,test_data=trainTestSplitLatest(modelling_table)
DA_Price,SS_Price=generateTestPricesLatest(IntegratedDataset)

#%% Training config
device='cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = NNDataset(features=features_train, labels=labels_train,powererrs=train_data["power_error"].values)
test_dataset = NNDataset(features=features_test, labels=labels_test,powererrs=test_data["power_error"].values)

train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MLPModel(input_size=features_train.shape[1], hidden_layer_size=32, output_size=1,num_layers=3).to(device)

model.load_state_dict(torch.load("models/Prices/latest/pre_train_model.pth"))
optimizer = optim.Adam(model.parameters(), lr=1e-3)


#%% Training Loop
model.eval()
model=train(100,train_dataloader,device,optimizer,model)

#Save model
torch.save(model.state_dict(), "models/Prices/latest/e2e_model.pth")

#%% Trading Revenue
model.eval()
pd_pred=model(torch.tensor(np.array(features_test,dtype=np.float32),dtype=torch.float32,device=device)).detach().squeeze().cpu().numpy()
biddings= test_data["power_pred"] + pd_pred
biddings[biddings<0]=0
biddings[biddings>1800]=1800
Revenue=utils.getRevenue(biddings,test_data["total_generation_MWh"],DA_Price,SS_Price)

R4=pd.read_csv("data/TradingRevenue.csv")
Revenue=Revenue.sum()+R4.values[0:2880].sum()
print("Revenue:",Revenue)




