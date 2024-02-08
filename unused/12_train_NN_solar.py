import numpy as np
import torch
import pandas as pd
import multiprocessing
from models import SolarMLP
from comp_utils import pinball
import os
import matplotlib.pyplot as plt
import pickle

class Dataset(torch.utils.data.Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.float32)
        
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
    
    def __len__(self):
        return len(self.features)


#随机数种子
torch.manual_seed(1)

#GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#超参数
hidden_size=16
num_epochs=100
learning_rate=1e-3
batch_size=256

#读取数据集
SolarDataset=pd.read_csv("data/dataset/SolarDataset.csv")


#提取features和labels
features=SolarDataset.iloc[:,:-1]
labels=SolarDataset.iloc[:,-1]

#z-score标准化
with open("data/dataset/Dataset_stats.pkl","rb") as f:
    Dataset_stats=pickle.load(f)

features=(features-Dataset_stats["Mean"]["features"]["solar"])/Dataset_stats["Std"]["features"]["solar"]
labels=(labels-Dataset_stats["Mean"]["labels"]["solar"])/Dataset_stats["Std"]["labels"]["solar"]

#转换为numpy数组
features=np.array(features)
labels=np.array(labels)

#划分训练集、验证集和测试集7:1:2
train_features=features[:int(0.7*len(features))]
train_labels=labels[:int(0.7*len(labels))]
val_features=features[int(0.7*len(features)):int(0.8*len(features))]
val_labels=labels[int(0.7*len(labels)):int(0.8*len(labels))]

#转换为Pytorch Dataset
train_dataset=Dataset(train_features,train_labels)
val_dataset=Dataset(val_features,val_labels)
full_dataset=Dataset(features,labels) 

#定义DataLoader
train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
val_dataloader=torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
full_dataloader=torch.utils.data.DataLoader(full_dataset,batch_size=batch_size,shuffle=True)

#初始化模型
Models_solar={}
for quantile in range(10,100,10):

    model_solar=SolarMLP(input_size=features.shape[1],
                hidden_size=hidden_size,
                output_size=1).to(device=device)
    
    Models_solar[f"q{quantile}"]=model_solar



#%% 训练模型
def train(quantile,full=False):

    global Models_wind,train_dataloader,val_dataloader,full_dataloader,device,num_epochs,learning_rate
    
    model=Models_solar[f"q{quantile}"]
    
    #定义损失函数为pinball loss
    criterion = lambda y,y_hat: pinball(y=y,y_hat=y_hat,alpha=quantile/100) #匿名函数

    #定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if full==True:
        dataloader=full_dataloader
    else:
        dataloader=train_dataloader

    train_loss=[]
    val_loss=[]

    for epoch in range(num_epochs):
        model.train()
        train_loss_tmp=0

        for i, (feature, label) in enumerate(dataloader):

            #数据转移到计算设备
            feature = feature.to(device=device)
            label = label.to(device=device).unsqueeze(1)

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
            for i, (feature, label) in enumerate(val_dataloader):
                
                #数据转移到计算设备
                feature = feature.to(device=device)
                label = label.to(device=device).unsqueeze(1)

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

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss_tmp:.4f}, Val Loss: {val_loss_tmp:.4f}')

        #将loss写入文件
        if not os.path.exists(f"training_log/solar_q{quantile}"):
            os.makedirs(f"training_log/solar_q{quantile}")

        with open(f"training_log/solar_q{quantile}/train_loss.txt", "a") as f:
            f.write(str(train_loss_tmp))
            f.write("\n")
        with open(f"training_log/solar_q{quantile}/val_loss.txt", "a") as f:
            f.write(str(val_loss_tmp))
            f.write("\n")

    #保存模型
    if full:
        torch.save(model.state_dict(), f"models/NN/full/solar_q{quantile}.pt")
    else:
        torch.save(model.state_dict(), f"models/NN/partial/solar_q{quantile}.pt")

def showLoss():

    #从文件中读取loss
    train_loss={}
    val_loss={}
    for quantile in range(10,100,10):
        train_loss[quantile]=[]
        val_loss[quantile]=[]
        with open(f"training_log/solar_q{quantile}/train_loss.txt", "r") as f:
            for line in f.readlines():
                train_loss[quantile].append(float(line))
            
        with open(f"training_log/solar_q{quantile}/val_loss.txt", "r") as f:
            for line in f.readlines():
                val_loss[quantile].append(float(line))

    #绘制loss曲线 3x3 subplot，展示每个quantile的loss曲线
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.plot(train_loss[(i+1)*10],label="train")
        plt.plot(val_loss[(i+1)*10],label="val")
        plt.title(f"q={(i+1)*10}%")
        plt.xlim(0,num_epochs+20)
        plt.ylim(0,0.15)
        plt.legend()
    plt.tight_layout()
    plt.show()



#%%
if __name__ == "__main__":

    #清空路径下的所有文件
    for quantile in range(10,100,10):
        for root, dirs, files in os.walk(f"training_log/solar_q{quantile}"):
            for name in files:
                os.remove(os.path.join(root, name))
                
    #选择是否全训练
    _input=input("是否全训练？(y/n)")
    if _input=="y":
        full=True
    elif _input=="n":
        full=False
    else:
        raise ValueError("输入错误")

    #创建多进程
    Process=[]
    for quantile in range(10,100,10):
        Process.append(multiprocessing.Process(target=train,args=(quantile,full)))

    #启动进程
    for p in Process:
        p.start()

    #等待进程结束
    for p in Process:
        p.join()

    #绘制loss曲线
    showLoss()

        
        
        
