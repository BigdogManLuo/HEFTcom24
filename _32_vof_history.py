import utils
from utils_vof import *
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
torch.manual_seed(0)

#%% Preparing dataset
IntegratedDataset, modelling_table=generateModellingTableHistory()
features_train,features_test,labels_train,labels_test,train_data,test_data=trainTestSplitHistory(modelling_table)
DA_Price,SS_Price=generateTestPricesHistory(IntegratedDataset)


#%% Training Configuration 
device='cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = NNDataset(features=features_train, labels=labels_train,powererrs=train_data["power_error"].values)
test_dataset = NNDataset(features=features_test, labels=labels_test,powererrs=test_data["power_error"].values)

train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True,drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

# Initialize the model, loss function, and optimizer
model = MLPModel(input_size=features_train.shape[1], hidden_layer_size=32, output_size=1,num_layers=10).to(device)
model.load_state_dict(torch.load("models/Prices/history/pre_train_model.pth"))
optimizer = optim.Adam(model.parameters(), lr=1e-5)

#%% Training Loop
model.eval()
model=train(50,train_dataloader,device,optimizer,model)

#Save model
torch.save(model.state_dict(), "models/Prices/history/e2e_model.pth")

#%% Trading Revenue
model.eval()
pd_pred=model(torch.tensor(np.array(features_test,dtype=np.float32),dtype=torch.float32,device=device)).detach().squeeze().cpu().numpy()
biddings= test_data["power_pred"] + pd_pred
biddings[biddings<0]=0
biddings[biddings>1800]=1800
Revenue=utils.getRevenue(biddings,test_data["total_generation_MWh"],DA_Price,SS_Price)
print("Revenue:",Revenue.sum())
