import pickle
from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
import multiprocessing
from comp_utils import pinball
import optuna
import matplotlib.pyplot as plt
import torch
import os

class Trainer():

    def __init__(self,type,source,Regressor,full,model_name):
        
        self.type=type
        self.source=source
        self.Regressor=Regressor
        self.full=full
        self.model_name=model_name

        if full==True:
            self.path="full"
        else:
            self.path="partial"
        
        #加载数据集
        self.dataset=pd.read_csv(f"data/dataset/{source}/{type.capitalize()}Dataset.csv")

        #提取features和labels
        self.features_table=self.dataset.iloc[:,:-1]
        self.labels_table=self.dataset.iloc[:,-1]

        #划分训练集和测试集
        self.train_features_table=self.features_table[int(0.35*len(self.features_table)):int(0.8*len(self.features_table))]
        self.train_labels_table=self.labels_table[int(0.35*len(self.labels_table)):int(0.8*len(self.labels_table))]
        self.test_features_table=self.features_table[int(0.8*len(self.features_table)):]
        self.test_labels_table=self.labels_table[int(0.8*len(self.labels_table)):]

        #转换为numpy
        self.train_features=np.array(self.train_features_table)
        self.train_labels=np.array(self.train_labels_table)
        self.test_features=np.array(self.test_features_table)
        self.test_labels=np.array(self.test_labels_table)

        #初始化模型
        self.Models={}

    def train(self,Params):
        if self.full==True:
            train_features=self.features[int(0.35*len(self.features)):]
            train_labels=self.labels[int(0.35*len(self.labels)):]
        else:
            train_features=self.train_features
            train_labels=self.train_labels

        #训练
        for quantile in range(10,100,10):
            self.Models[f"q{quantile}"]=self.Regressor(**Params[f"q{quantile}"])
            self.Models[f"q{quantile}"].fit(train_features,train_labels)

        #保存
        for quantile in range(10,100,10):
            with open(f"models/{self.model_name}/{self.path}/{self.type}_q{quantile}.pkl","wb") as f:
                pickle.dump(self.Models[f"q{quantile}"],f)
                
    def test(self):
        
        #加载模型
        Models={}
        for quantile in range(10,100,10):
            with open(f"models/{self.model_name}/{self.path}/{self.type}_q{quantile}.pkl","rb") as f:
                Models[f"q{quantile}"]=pickle.load(f)

        #测试
        predictions={}
        Pinball_Loss={}
        for quantile in range(10,100,10):
            #前向
            y_hat=Models[f"q{quantile}"].predict(self.test_features)

            #计算损失
            loss=pinball(y=self.test_labels,y_hat=y_hat,alpha=quantile/100)

            #保存
            predictions[f"q{quantile}"]=y_hat
            Pinball_Loss[f"q{quantile}"]=loss

        #计算平均得分
        Pinball_Loss_mean=np.array(list(Pinball_Loss.values())).mean()
        print("Score:",Pinball_Loss_mean)
        
        #展示50%分位数预测-真实值散点图
        plt.scatter(self.test_labels,predictions["q50"],color="blue",s=10,alpha=0.3)
        plt.plot(self.test_labels,self.test_labels,color="red",linestyle="--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.show()
        
            
    def kfold_train(self,Params,num_folds,quantile):
        
        kf = KFold(n_splits=num_folds)
        predictions =[]

        #将训练数据划分为多份
        for train_index, val_index in kf.split(self.train_features):

            #在当前份上训练
            train_features=self.train_features[train_index]
            train_labels=self.train_labels[train_index]
            val_features=self.train_features[val_index]
            model=self.Regressor(**Params)
            model.fit(train_features,train_labels)

            #在当前份上预测
            prediction=model.predict(val_features)
                
            #合并
            predictions.append(prediction)

        predictions=np.concatenate(predictions,axis=0)

        #保存结果，如果路径不存在则创建
        if not os.path.exists(f"predictions/{self.model_name}"):
            os.makedirs(f"predictions/{self.model_name}")
        np.save(f"predictions/{self.model_name}/{self.type}_q{quantile}.npy",predictions)

        #保存模型，如果路径不存在则创建
        if not os.path.exists(f"models/stacking/{self.model_name}"):
            os.makedirs(f"models/stacking/{self.model_name}")
        with open(f"models/stacking/{self.model_name}/{self.type}_q{quantile}.pkl","wb") as f:
            pickle.dump(model,f)

    def kfold_train_parallel(self,Params,num_folds):
        #创建多进程
        Process=[]
        for quantile in range(10,100,10):
            Process.append(multiprocessing.Process(target=self.kfold_train,args=(Params,num_folds,quantile)))
            
        #启动进程
        for p in Process:
            p.start()

        #等待进程结束
        for p in Process:
            p.join()
                

class Dataset(torch.utils.data.Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.float32)
        
    def __getitem__(self,index):
        return self.features[index],self.labels[index]
    
    def __len__(self):
        return len(self.features)


class NNRegressor():
    def __init__(self,lr,batch_size,num_epochs,hidden_size,loss_fn,alpha,num_layers,verbose=False):
        
        self.lr=lr
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.hidden_size=hidden_size
        self.loss_fn=loss_fn
        self.verbose=verbose
        self.num_layers=num_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.alpha=alpha
        if loss_fn=="quantile":
            self.criterion = pinball
        elif loss_fn=="mse":
            self.criterion = torch.nn.MSELoss()
        elif loss_fn=="mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise ValueError("loss_fn must be one of ['quantile','mse','mae']")

        self.Dataset_stats={}

    def fit(self,features,labels):
        
        '''
        #归一化
        features=(features-features.mean(axis=0))/features.std(axis=0)
        labels=(labels-labels.mean())/labels.std()
        
        #保存归一化数据到属性
        self.Dataset_stats={"Mean":{"features":features.mean(axis=0),"labels":labels.mean(axis=0)},"Std":{"features":features.std(axis=0),"labels":labels.std(axis=0)}}
        '''

        #创建Dataset和DataLoader
        dataset=Dataset(features,labels)
        dataloader=torch.utils.data.DataLoader(dataset,batch_size=self.batch_size,shuffle=True)

        #构造模型
        input_size=features.shape[1]
        if len(labels.shape)==1:
            output_size=1
        else:
            output_size=labels.shape[1]
        
        self.model=torch.nn.Sequential()
        self.model.add_module("input",torch.nn.Linear(input_size,self.hidden_size))
        self.model.add_module("relu",torch.nn.ReLU())
        for i in range(self.num_layers-1):
            self.model.add_module(f"hidden_{i}",torch.nn.Linear(self.hidden_size,self.hidden_size))
            self.model.add_module(f"relu_{i}",torch.nn.ReLU())
        self.model.add_module("output",torch.nn.Linear(self.hidden_size,output_size))
        self.model.to(device=self.device)

        #定义优化器
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss_tmp=0
            for i, (feature, label) in enumerate(dataloader):

                #数据转移到计算设备
                feature = feature.to(device=self.device)
                label = label.to(device=self.device).unsqueeze(1)

                #前向传播
                outputs = self.model(feature)

                #计算损失
                if outputs.shape!=label.shape:
                    raise ValueError("outputs shape is not equal to label shape")
                else:
                    loss = self.criterion(y_hat=outputs,y=label,alpha=self.alpha)

                #反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                #记录损失
                train_loss_tmp+=loss.item()

            train_loss_tmp=train_loss_tmp/(i+1)

            if self.verbose==True:
                print(f"Epoch:{epoch+1}\tLoss:{train_loss_tmp}")
        
    def predict(self,features):
        self.model.eval()
        features=torch.tensor(features,dtype=torch.float32).to(device=self.device)
        
        #预测
        y_hat=self.model(features)

        #反归一化
        #y_hat=y_hat.detach().cpu().numpy().flatten()*self.Dataset_stats["Std"]["labels"]+self.Dataset_stats["Mean"]["labels"]

        return y_hat.detach().cpu().numpy().flatten()
    
    def save(self,path):

        #保存模型
        torch.save(self.model.state_dict(),path)


class HyperParamsOptimizer():
    
    def __init__(self,train_features,train_labels,Regressor,params_raw):
        self.train_features=train_features
        self.train_labels=train_labels
        self.Regressor=Regressor
        self.params_raw=params_raw
    
    def objective(self,trial,quantile):
            
        #交叉验证
        cv = KFold(n_splits=4, shuffle=True, random_state=2048)
        cv_scores = np.empty(4)
        for idx, (train_idx, test_idx) in enumerate(cv.split(self.train_features, self.train_labels)):
            X_train, X_test = self.train_features[train_idx], self.train_features[test_idx]
            y_train, y_test = self.train_labels[train_idx], self.train_labels[test_idx]

            model=self.Regressor(**self.params_raw)
            model.fit(X_train,y_train,eval_set=[(X_test,y_test)])
            #在当前份上预测
            preds=model.predict(X_test)
            
            cv_scores[idx] = pinball(y_test,preds,alpha=quantile/100)
            
        return np.mean(cv_scores)
    
    def optuna_train(self,quantil):
        study = optuna.create_study(direction="minimize")
        func = lambda trial: self.objective(trial,quantil)
        study.optimize(func, n_trials=20)
        print(f"\tquantil:{quantil}\tBest params:")
        for key, value in study.best_params.items():
            print(f"\t\t{key}: {value}")
            
        #保存
        with open(f"best_params/{self.model_name}/{self.type}_q{quantil}.pkl","wb") as f:
            pickle.dump(study.best_params,f)
            
    def optuna_train_parallel(self):
        #创建多进程
        Process=[]
        for quantile in range(10,100,10):
            Process.append(multiprocessing.Process(target=self.optuna_train,args=(quantile,)))
        
        #启动进程
        for p in Process:
            p.start()

        #等待进程结束
        for p in Process:
            p.join()
