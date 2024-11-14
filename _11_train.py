from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from utils import Trainer
import optuna
from itertools import chain
from tqdm import tqdm

def load_best_params(target):

    best_params={}  
    for quantile in range(10,100,10):
        study = optuna.load_study(study_name=f"Predictor_{target}_q{quantile}", storage=f"sqlite:///data/best_params/{target}_q{quantile}.db")
        best_params[f"q{quantile}"]=study.best_params
        best_params[f"q{quantile}"]["objective"]="quantile"
        best_params[f"q{quantile}"]["alpha"]=quantile/100
        best_params[f"q{quantile}"]["verbose"]=-1

    return best_params

def getBestParams():

    Params_wind_forecast={}
    Params_solar_forecast={}

    for quantile in chain([0.1],range(1,100,1),[99.9]):
    
        params={
            'objective':'quantile',
            'alpha':quantile/100,
            'num_leaves': 500,
            'n_estimators': 2000,
            'max_depth':6,
            'min_data_in_leaf': 200,
            'learning_rate':0.2,
            'lambda_l1': 40,           
            'lambda_l2': 80,
            'verbose':-1
            }        
        Params_wind_forecast[f"q{quantile}"]=params

        params={
            'objective':'quantile',
            'alpha':quantile/100,
            'num_leaves': 100,
            'n_estimators': 2200,
            'max_depth':10,
            'min_data_in_leaf': 1800,
            'learning_rate':0.19,
            'lambda_l1': 20,           
            'lambda_l2': 40,
            'verbose':-1
            }
        Params_solar_forecast[f"q{quantile}"]=params

    return Params_wind_forecast,Params_solar_forecast


#----------------------Hyperparams----------------------
params_LGBM_wind_forecast, params_LGBM_solar_forecast=getBestParams()

params_LGBM_wind_trading={
    'objective':'mse',
    'num_leaves': 1000,
    'n_estimators': 500,
    'max_depth':6,
    'min_data_in_leaf': 700,
    'learning_rate':0.078,
    'lambda_l1': 70,
    'lambda_l2': 40,
    'verbose':-1
    }

params_LGBM_solar_trading={
    'objective':'mse',
    'num_leaves': 700,
    'n_estimators': 2000,
    'max_depth':9,
    'min_data_in_leaf': 1400,
    'learning_rate':0.063,
    'lambda_l1': 80,           
    'lambda_l2': 40,
    'verbose':-1}


if __name__=="__main__":

    for source in ["dwd","gfs"]:
        
        for is_full in [False]:
    
            for target in ["solar","wind"]:
                
                configs={
                    "target_type":target,
                    "Regressor":LGBMRegressor,
                    "full":is_full, 
                    "model_name":"LGBM", 
                    "source":source 
                    }
                trainer=Trainer(**configs)
                
                if target=="wind":
                    trainer.train(params_LGBM_wind_forecast)
                    trainer.train_bidding(params_LGBM_wind_trading)
                else:
                    trainer.train(params_LGBM_solar_forecast)
                    trainer.train_bidding(params_LGBM_solar_trading)
            
        
        
        
        
        
        
        
        
