from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from utils import Trainer,HyperParamsOptimizer,NNRegressor
import dill as pickle
from statsmodels.iolib import smpickle
import numpy as np
#import pickle
from tqdm import tqdm
import optuna
import pandas as pd

def defineParams(regressor_name,target_type):

    if regressor_name=="LGBM":
        
        Params={}
        if target_type=="solar":
            
            Num_leaves=[350,350,350,350,600,475,475,400,150]
            Num_estimators=[550,550,550,400,1000,1000,385,300,650]
            Min_data_in_leaf=[300,300,200,300,200,200,400,200,200]
            l2=[5,5,5,5,0,30,30,20,50]

        elif target_type=="wind":

            Num_leaves=[100,100,250,425,600,425,100,275,250]
            Num_estimators=[100,100,100,100,100,100,200,100,100]
            Min_data_in_leaf=[700,300,1000,1000,800,1000,300,900,900]
            l2=[30,0,50,50,20,35,0,30,40]
        
        for idx, quantile in enumerate(range(10,100,10)):
            params={
                'objective': 'quantile',
                'alpha':quantile/100,
                'num_leaves': Num_leaves[idx],
                'n_estimators': Num_estimators[idx],
                'min_data_in_leaf': Min_data_in_leaf[idx],
                'lambda_l2':l2[idx],
                'verbose':-1
                }
            
            Params[f"q{quantile}"]=params

    elif regressor_name=="CatBoost":
        Params={}
        for idx, quantile in enumerate(range(10,100,10)):
            params={
                'iterations':250, 
                'learning_rate':1e-1,
                'silent':True,
                'loss_function':f'Quantile:alpha={quantile/100}'
                }
            
            Params[f"q{quantile}"]=params
            

    elif regressor_name=="MLP":
        Params={}
        for idx, quantile in enumerate(range(10,100,10)):
            params = {
                "lr": 0.05,
                "batch_size": 1024,
                "num_epochs": 50,
                "hidden_size": 64,
                "num_layers": 3,
                "loss_fn": "quantile",
                "alpha": quantile/100
                }
            
            Params[f"q{quantile}"]=params
        

    return params


if __name__ == "__main__":
    
   
    for target_type in ["wind","solar"]:
        
        #==========================================训练器配置=================================
        Regressors=[LGBMRegressor,CatBoostRegressor]
        Regressor_names=["LGBM","CatBoost"]
        
        
        Trainers={}
        for regressor_name,regressor in zip(Regressor_names,Regressors):
            configs={
            "source":"dwd", #数据来源
            "type":target_type, #预测对象类型
            "Regressor":regressor, #模型类型
            "full":False, #是否使用全量数据
            "model_name":regressor_name #模型名称
            }
            Trainers[regressor_name]=Trainer(**configs)
        
        #===============================训练基学习器=====================================
        for regressor_name in tqdm(Regressor_names):
            
            Params=defineParams(regressor_name,target_type)
            Trainers[regressor_name].kfold_train_parallel(Params,num_folds=2)    


        #===============================训练元学习器=====================================
        for quantile in range(10,100,10):
            
            #加载基学习器预测结果
            Predictions=[]
            for regressor_name in Regressor_names:
                Predictions.append(np.load(f"predictions/{regressor_name}/{target_type}_q{quantile}.npy"))

            #加载Benchmark模型的预测结果
            with open(f"models/benchmark/{target_type}_q{quantile}.pickle", 'rb') as f:
                mod_benchmark= smpickle.load_pickle(f)

            if target_type=="solar":

                features_table=Trainers[Regressor_names[0]].train_features_table["rad_t_dwd"].to_frame()            
                features_table.rename(columns={"rad_t_dwd":"SolarDownwardRadiation"},inplace=True)
            elif target_type=="wind":

                features_table=Trainers[Regressor_names[0]].train_features_table["ws_100_t_dwd_1"].to_frame()
                features_table.rename(columns={"ws_100_t_dwd_1":"WindSpeed"},inplace=True)
                
            predictions=np.array(mod_benchmark.predict(features_table))
            Predictions.append(predictions)

            #创建数据集
            Features=np.stack(Predictions,axis=-1)
            Labels=Trainers[Regressor_names[0]].train_labels


            #参数定义
            params={
                "lr": 0.01,
                "batch_size": 1024,
                "num_epochs": 20,
                "hidden_size": 32,
                "num_layers": 1,
                "loss_fn": "quantile",
                "alpha": quantile/100
                }
            
            #训练
            model=NNRegressor(**params)
            model.fit(Features,Labels)

            #保存模型
            with open(f"models/stacking/meta/{target_type}_q{quantile}.pkl","wb") as f:
                pickle.dump(model,f)
    
    
    
    
    
    