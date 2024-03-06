from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor
from utils import Trainer,HyperParamsOptimizer
from statsmodels.iolib import smpickle
import numpy as np
import pickle
from tqdm import tqdm
import optuna
import pandas as pd

def defineParams(regressor_name):

    if regressor_name=="LGBM":

        params={
            'objective': 'mae',
            'num_leaves': 600,
            'n_estimators':1000,
            'min_data_in_leaf': 200,
            'lambda_l2':0,
            'verbose':-1,

            }

    elif regressor_name=="CatBoost":
        
        params={
            'iterations':200, 
            'learning_rate':1e-1,
            'silent':True,
            'loss_function':'Quantile:alpha=0.5'
            }

    elif regressor_name=="MLP":

        params={
            "hidden_layer_sizes":(32,32,32),
            "max_iter":50,
            "verbose":False,
        }

    elif regressor_name=="ExtraTrees":

        params={
            "n_estimators":300,
            "max_depth":8,
            "min_samples_split":5,
            "min_samples_leaf":1,
            "verbose":0,
        }

    return params


if __name__ == "__main__":
    
    for target_type in ["wind","solar"]:
        #训练器配置
        Regressors=[LGBMRegressor,CatBoostRegressor,MLPRegressor,ExtraTreesRegressor]
        Regressor_names=["LGBM","CatBoost","MLP","ExtraTrees"]
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
        
            #分别训练每个模型(k-fold)
            Params=defineParams(regressor_name)
            Trainers[regressor_name].kfold_train(Params,num_folds=3)    
    
        #按顺序加载每个元学习器的预测结果
        Predictions=[]
        for regressor_name in Regressor_names:
            Predictions.append(np.load(f"predictions/{regressor_name}/"+configs["type"]+".npy"))
        
        '''
        #加载Benchmark模型的预测结果
        with open(f"models/benchmark/{target_type}_q50.pickle", 'rb') as f:
            mod_benchmark= smpickle.load_pickle(f)

        if target_type=="solar":
            features_table=Trainers[Regressor_names[0]].train_features_true["rad_t_dwd"].to_frame()
            features_table.rename(columns={"rad_t_dwd":"SolarDownwardRadiation"},inplace=True)
        elif target_type=="wind":
            features_table=Trainers[Regressor_names[0]].train_features_true["ws_100_t_dwd_1"].to_frame()
            features_table.rename(columns={"ws_100_t_dwd_1":"WindSpeed"},inplace=True)

        predictions=np.array(mod_benchmark.predict(features_table))
        Predictions.append(predictions)
        '''
        
        #创建数据集
        Features=np.stack(Predictions,axis=-1)
        Labels=Trainers[Regressor_names[0]].train_labels_true
    
        #训练元学习器    
        for idx, quantile in enumerate(range(10,100,10)):
            
            params={
                'objective': 'quantile',
                'num_leaves': 7,
                'n_estimators': 50,
                'max_depth':10,
                'lambda_l2':2,
                'verbose':-1,
                'alpha':quantile/100
                }
            
            Params[f"q{quantile}"]=params
    
            model=LGBMRegressor(**params)
            model.fit(Features,Labels)
    
            #保存模型
            with open("models/stacking/meta/"+configs["type"]+f"_q{quantile}.pkl","wb") as f:
                pickle.dump(model,f)




    

        



        
   

    
    
    
    
    
    
    
    
    
    
    
    