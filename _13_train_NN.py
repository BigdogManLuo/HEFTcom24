import numpy as np
import torch
import pandas as pd
import multiprocessing
from comp_utils import pinball
import os
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
from sklearn.model_selection import KFold
from utils import Trainer

class Dataset(torch.utils.data.Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.float32)
        
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
    
    def __len__(self):
        return len(self.features)

class SolarMLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(SolarMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out
    
class WindMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(WindMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.relu = nn.ReLU()  
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


class NNTrainer(Trainer):

    def __init__(self,type,source,Params):
        
        super().__init__(type,source)
        torch.manual_seed(1)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs=Params['q10']["num_epochs"]
        self.learning_rate=Params['q10']["learning_rate"]
        self.batch_size=Params['q10']["batch_size"]

        self.type=type
        #加载原数据集
        if type=="wind":
            self.dataset=pd.read_csv(f"data/dataset/{source}/WindDataset.csv")
        elif type=="solar":
            self.dataset=pd.read_csv(f"data/dataset/{source}/SolarDataset.csv") 

        #转换为Dataloader
        train_dataset=Dataset(self.train_features,self.train_labels)
        val_dataset=Dataset(self.test_features,self.test_labels)
        full_dataset=Dataset(self.features,self.labels)
        self.train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)
        self.val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=self.batch_size,shuffle=True)
        self.full_dataloader=torch.utils.data.DataLoader(full_dataset,batch_size=self.batch_size,shuffle=True)

        #初始化模型
        self.Models={}
        if type=="wind":
            for quantile in range(10,100,10):
                self.Models[f"q{quantile}"]=WindMLP(input_size=self.features.shape[1],
                    hidden_size=Params[f"q{quantile}"]["hidden_size"],
                    output_size=1).to(device=self.device)
            
        elif type=="solar":
            for quantile in range(10,100,10):
                self.Models[f"q{quantile}"]=SolarMLP(input_size=self.features.shape[1],
                    hidden_size=Params[f"q{quantile}"]["hidden_size"],
                    output_size=1).to(device=self.device)

        
    def train(self,quantile,full=False):

        if full==True:
            dataloader=self.full_dataloader
        else:
            dataloader=self.train_dataloader

        #获取模型
        model=self.Models[f"q{quantile}"]

        #定义损失函数为pinball loss
        criterion = lambda y,y_hat: pinball(y=y,y_hat=y_hat,alpha=quantile/100) #匿名函数

        #定义优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        train_loss=[]
        val_loss=[]

        for epoch in range(self.num_epochs):
            model.train()
            train_loss_tmp=0

            for i, (feature, label) in enumerate(dataloader):

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
            train_loss.append(train_loss_tmp)
            
            model.eval()
            val_loss_tmp=0
            with torch.no_grad():
                for i, (feature, label) in enumerate(self.val_dataloader):
                    
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

                    #记录损失
                    val_loss_tmp+=loss.item()

                val_loss_tmp=val_loss_tmp/(i+1)
                val_loss.append(val_loss_tmp)

            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss_tmp:.4f}, Val Loss: {val_loss_tmp:.4f}')

            #将loss写入文件
            if not os.path.exists(f"training_log/{self.type}_q{quantile}"):
                os.makedirs(f"training_log/{self.type}_q{quantile}")

            with open(f"training_log/{self.type}_q{quantile}/train_loss.txt", "a") as f:
                f.write(f"{train_loss_tmp}\n")
            with open(f"training_log/{self.type}_q{quantile}/val_loss.txt", "a") as f:
                f.write(f"{val_loss_tmp}\n")

        #保存模型
        if full:
            torch.save(model.state_dict(), f"models/NN/full/{self.type}_q{quantile}.pt")
        else:
            torch.save(model.state_dict(), f"models/NN/partial/{self.type}_q{quantile}.pt")

    def train_parallel(self,quantile,full):
        #创建多进程
        Process=[]
        for quantile in range(10,100,10):
            Process.append(multiprocessing.Process(target=self.train,args=(quantile,full)))
        
        #启动进程
        for p in Process:
            p.start()

        #等待进程结束
        for p in Process:
            p.join()
    
    
    def kflod_train_single(self,quantile,num_folds):

        kf = KFold(n_splits=num_folds)
        predictions =[]

        for train_index, val_index in kf.split(self.train_features):
            
            #获取当前份的数据
            train_features=self.train_features[train_index]
            train_labels=self.train_labels[train_index]
            val_features=self.train_features[val_index]
            val_labels=self.train_labels[val_index]

            #加载到dataloader
            train_dataset=Dataset(train_features,train_labels)
            val_dataset=Dataset(val_features,val_labels)
            train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size,shuffle=True)
            val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=1024,shuffle=True,drop_last=False)

            
            #获得一个初始化的模型
            model=self.Models[f"q{quantile}"]

            #在当前份上训练
            criterion = lambda y,y_hat: pinball(y=y,y_hat=y_hat,alpha=quantile/100)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            for epoch in range(self.num_epochs):
                model.train()

                for i, (feature, label) in enumerate(train_dataloader):

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
            
            #在当前份上预测
            model.eval()
            with torch.no_grad():
                for i, (feature, label) in enumerate(val_dataloader):
                    feature = feature.to(device=self.device)
                    outputs = model(feature)
                    #反归一化
                    outputs=outputs.cpu().numpy()*self.Dataset_stats["Std"]["labels"][self.type]+self.Dataset_stats["Mean"]["labels"][self.type]
                    predictions.append(outputs)

        predictions=np.concatenate(predictions)

        #保存预测结果
        np.save(f"predictions/NN/{self.type}_q{quantile}.npy",predictions)

    def k_fold_train(self,num_folds):
        #创建多进程
        Process=[]
        for quantile in range(10,100,10):
            Process.append(multiprocessing.Process(target=self.kflod_train_single,args=(quantile,num_folds)))
        
        #启动进程
        for p in Process:
            p.start()

        #等待进程结束
        for p in Process:
            p.join()
            
        #加载所有预测结果
        predictions = {}
        for quantile in range(10,100,10):
            predictions[f"q{quantile}"]=np.load(f"predictions/NN/{self.type}_q{quantile}.npy")
            
        return predictions
        
    def showLoss(self):

        #从文件中读取loss
        train_loss={}
        val_loss={}
        for quantile in range(10,100,10):
            train_loss[quantile]=[]
            val_loss[quantile]=[]
            with open(f"training_log/{type}_q{quantile}/train_loss.txt", "r") as f:
                for line in f.readlines():
                    train_loss[quantile].append(float(line))
            with open(f"training_log/{type}_q{quantile}/val_loss.txt", "r") as f:
                for line in f.readlines():
                    val_loss[quantile].append(float(line))

        #绘制loss曲线 3x3 subplot，展示每个quantile的loss曲线
        plt.figure(figsize=(10,10))
        for i in range(9):
            plt.subplot(3,3,i+1)
            plt.plot(train_loss[(i+1)*10],label="train")
            plt.plot(val_loss[(i+1)*10],label="val")
            plt.title(f"q={(i+1)*10}%")
            plt.xlim(0,self.num_epochs+20)
            plt.ylim(0,0.5)
            plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    
    #预测模型类型
    type="wind"
    
    source="dwd"
    
    #选择是否全训练
    full= False
    
    #超参数
    Params={}
    for quantile in range(10,100,10):
        params={
            "num_epochs":15,
            "batch_size":512,
            "learning_rate":1e-3,
            "hidden_size":32
        }
        Params[f"q{quantile}"]=params
        
        

    #清空路径下的所有文件
    for quantile in range(10,100,10):
        for root, dirs, files in os.walk(f"training_log/{type}_q{quantile}"):
            for name in files:
                os.remove(os.path.join(root, name))
    
    #创建训练器
    trainer=NNTrainer(type=type,source=source,Params=Params)
    
    #训练
    trainer.kflod_train_single(quantile=10,num_folds=3)
    
    #trainer.showLoss()
    



