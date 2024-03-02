import pickle
from typing import Any
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import multiprocessing
from comp_utils import pinball
import optuna
import matplotlib.pyplot as plt

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
        if type=="wind":
            self.dataset=pd.read_csv(f"data/dataset/{source}/WindDataset.csv")
        elif type=="solar":
            self.dataset=pd.read_csv(f"data/dataset/{source}/SolarDataset.csv")

        #提取features和labels
        self.features=self.dataset.iloc[:,:-1]
        self.labels=self.dataset.iloc[:,-1]

        #z-score标准化
        with open(f"data/dataset/{source}/Dataset_stats.pkl","rb") as f:
            self.Dataset_stats=pickle.load(f)

        if type=="solar":
            self.features.iloc[:,0:-1]=(self.features.iloc[:,0:-1]-self.Dataset_stats["Mean"]["features"][type])/self.Dataset_stats["Std"]["features"][type] #最后一列是hour不需要标准化
        
        elif type=="wind":
            self.features=(self.features-self.Dataset_stats["Mean"]["features"][type])/self.Dataset_stats["Std"]["features"][type]
        
        self.labels=(self.labels-self.Dataset_stats["Mean"]["labels"][type])/self.Dataset_stats["Std"]["labels"][type]
        
        #转换为numpy数组
        self.features=np.array(self.features)
        self.labels=np.array(self.labels)
        
        #划分训练集和测试集
        self.train_features=self.features[int(0.35*len(self.features)):int(0.8*len(self.features))]
        self.train_labels=self.labels[int(0.35*len(self.labels)):int(0.8*len(self.labels))]
        self.test_features=self.features[int(0.8*len(self.features)):]
        self.test_labels=self.labels[int(0.8*len(self.labels)):]
        
        #初始化模型
        self.Models={}

        
    def train(self,Params):
        if self.full==True:
            train_features=self.features[int(0.35*len(self.features)):int(0.95*len(self.features))]
            train_labels=self.labels[int(0.35*len(self.labels)):int(0.95*len(self.features))]
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
        
        test_features=self.test_features
        test_labels=self.test_labels

        #label 反归一化
        test_labels=test_labels*self.Dataset_stats["Std"]["labels"][self.type]+self.Dataset_stats["Mean"]["labels"][self.type]

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
            y_hat=Models[f"q{quantile}"].predict(test_features)

            #反归一化
            y_hat=y_hat*self.Dataset_stats["Std"]["labels"][self.type]+self.Dataset_stats["Mean"]["labels"][self.type]

            #计算损失
            loss=pinball(y=test_labels,y_hat=y_hat,alpha=quantile/100)

            #保存
            predictions[f"q{quantile}"]=y_hat
            Pinball_Loss[f"q{quantile}"]=loss

        #计算平均得分
        Pinball_Loss_mean=np.array(list(Pinball_Loss.values())).mean()
        print("Score:",Pinball_Loss_mean)
        
        #展示50%分位数预测-真实值散点图
        plt.scatter(test_labels,predictions["q50"],color="blue",s=10,alpha=0.3)
        plt.plot(test_labels,test_labels,color="red",linestyle="--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.show()
        
            
    def kfold_train(self,Params,num_folds):
        
        kf = KFold(n_splits=num_folds)
        predictions = {f"q{quantile}": [] for quantile in range(10,100,10)}
        for quantile in range(10,100,10):

            Params[f"q{quantile}"]['alpha']=quantile/100

            #将训练数据划分为多份
            for train_index, val_index in kf.split(self.train_features):

                #在当前份上训练
                train_features=self.train_features[train_index]
                train_labels=self.train_labels[train_index]
                val_features=self.train_features[val_index]
                model=self.Regressor(**Params[f"q{quantile}"])
                model.fit(train_features,train_labels)

                #在当前份上预测
                prediction=model.predict(val_features)

                #反归一化
                prediction=prediction*self.Dataset_stats["Std"]["labels"][self.type]+self.Dataset_stats["Mean"]["labels"][self.type]
                
                #合并
                predictions[f"q{quantile}"].append(prediction)

            predictions[f"q{quantile}"]=np.concatenate(predictions[f"q{quantile}"])
        
        np.save(f"predictions/{self.model_name}/{self.type}_q{quantile}.npy",predictions)

    
    def kfold_train_parallel(self,num_folds):
        #创建多进程
        Process=[]
        for quantile in range(10,100,10):
            Process.append(multiprocessing.Process(target=self.kflod_train,args=(quantile,num_folds)))
            
        #启动进程
        for p in Process:
            p.start()

        #等待进程结束
        for p in Process:
            p.join()
            
        #加载所有预测结果
        predictions = {}
        for quantile in range(10,100,10):
            predictions[f"q{quantile}"]=np.load(f"predictions/{self.model_name}/{self.type}_q{quantile}.npy")
            
        return predictions



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
