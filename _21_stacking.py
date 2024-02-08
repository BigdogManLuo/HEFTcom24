import numpy as np
import pandas as pd
import pickle
from _11_train_LGBM import LGBMTrainer
from _13_train_NN  import NNTrainer
from _12_train_CatBoost import CatBoostTrainer
from sklearn.tree import DecisionTreeRegressor
from _13_train_NN import Dataset
import torch
import torch.nn as nn
from comp_utils import pinball

class MetaMLP(nn.Module):
    
    def __init__(self, input_size):
        super(MetaMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    

class MetaRegressor():
    
    def __init__(self,type,num_estimators,Features,Labels,params):
        
        self.type=type
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.model=MetaMLP(num_estimators)
            
        self.features=Features
        self.labels=Labels
        
        self.lr=params["lr"]
        self.batch_size=params["batch_size"]
        self.num_epochs=params["num_epochs"]
        
        dataset=Dataset(self.features,self.labels)
        self.dataloader=torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)
        
    def train(self,quantile):
        
        model=self.model
        
        #定义损失函数为pinball loss
        criterion = lambda y,y_hat: pinball(y=y,y_hat=y_hat,alpha=quantile/100) #匿名函数

        #定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            model.train()
            train_loss_tmp=0

            for i, (feature, label) in enumerate(self.dataloader):

                #数据转移到计算设备
                feature = feature.to(device=self.device)
                label = label.to(device=self.device).unsqueeze(1)

                #前向传播
                outputs = model(feature)

                #计算损失
                if outputs.shape!=label.shape:
                    raise ValueError("outputs shape is not equal to label shape")
                else:
                    loss = criterion(y_hat=outputs,y=label)

                #反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #记录损失
                train_loss_tmp+=loss.item()

            train_loss_tmp=train_loss_tmp/(i+1)
            
            print(f"Epoch:[{epoch}/{self.num_epochs}]  train_loss:{train_loss_tmp}")
        
        #保存模型
        torch.save(model.state_dict(), f"models/stacking/{self.type}_q{quantile}.pt")

if __name__ == "__main__":

    
    type="wind"
    source="dwd"
    full=False
    
    Params_lgbm={}
    Params_catboost={}
    Params_NN={}
    
    for idx, quantile in enumerate(range(10,100,10)):
        #超参数定义
        params_lgbm = {
            'objective': 'quantile',
            'num_leaves': 150,
            'n_estimators': 150,
            'min_data_in_leaf':50
            }
    
        params_catboost = {
            'iterations':300, 
            'learning_rate':1e-1
            }
        
        params_nn={
            "num_epochs":15,
            "batch_size":512,
            "learning_rate":1e-3,
            "hidden_size":32
            }
        
        
        Params_lgbm[f"q{quantile}"]=params_lgbm
        Params_catboost[f"q{quantile}"]=params_catboost
        Params_NN[f"q{quantile}"]=params_nn
    
    params_meta={
        'lr':5e-2,
        'batch_size':512,
        'num_epochs':20
    }
    
    #初始化训练器
    lgbm_trainer=LGBMTrainer(type,source)
    catboost_trainer=CatBoostTrainer(type,source)
    nn_trainer=NNTrainer(type,source,Params_NN)

    #分别执行k-fold训练
    Predictions=[]
    Predictions.append(lgbm_trainer.kfold_train(Params_lgbm,num_folds=3))
    Predictions.append(catboost_trainer.kfold_train(Params_catboost,num_folds=3))
    Predictions.append(nn_trainer.k_fold_train(num_folds=3))
    
    #训练元学习器
    for quantile in range(10,100,10):
        
        Features=np.stack([elem[f"q{quantile}"] for elem in Predictions],axis=-1)
        Labels=lgbm_trainer.train_labels*lgbm_trainer.Dataset_stats["Std"]["labels"][type]+lgbm_trainer.Dataset_stats["Mean"]["labels"][type]
        #Features=np.column_stack((lgbm_trainer.label_hours,Features))
        
        #定义元学习器
        model=MetaRegressor(type=type,num_estimators=len(Predictions),Features=Features,Labels=Labels,params=params_meta)

        #训练
        model.train(quantile)

        
   

    
    
    
    
    
    
    
    
    
    
    
    