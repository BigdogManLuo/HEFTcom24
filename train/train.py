import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMRegressor
#from catboost import CatBoostRegressor
from itertools import chain
import pandas as pd
import pickle
from tqdm import tqdm

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
            'random_state':42,
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
            'random_state':42,
            'verbose':-1
            }
        Params_solar_forecast[f"q{quantile}"]=params

    return Params_wind_forecast,Params_solar_forecast

class Trainer():

    def __init__(self,target_type,Regressor,full,model_name,source):

        self.type=target_type
        self.Regressor=Regressor
        self.full=full
        self.model_name=model_name
        self.source=source
        if full==True:
            self.path="full"
        else:
            self.path="train"
        
        #Load dataset
        self.dataset=pd.read_csv(f"../data/dataset/{self.path}/{source}/{self.type.capitalize()}Dataset.csv")
        self.features=self.dataset.iloc[:,:-1]
        self.labels=self.dataset.iloc[:,-1]

        #Delineating the scope of training
        if self.type=="wind":
            self.features=self.features
            self.labels=self.labels
        
        elif self.type=="solar":
            self.features=self.features[int(0.2*len(self.features)):]
            self.labels=self.labels[int(0.2*len(self.labels)):] 
        
        #Initialize models
        self.Models={}
        
    def train(self,Params):

        if not os.path.exists(f"../models/{self.model_name}/{self.path}/{self.source}"):
            os.makedirs(f"../models/{self.model_name}/{self.path}/{self.source}")

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            
            self.Models[f"q{quantile}"]=self.Regressor(**Params[f"q{quantile}"])
            self.Models[f"q{quantile}"].fit(self.features,self.labels)
            with open(f"../models/{self.model_name}/{self.path}/{self.source}/{self.type}_q{quantile}.pkl","wb") as f:
                pickle.dump(self.Models[f"q{quantile}"],f)

    def train_bidding(self,params):
        if not os.path.exists(f"../models/{self.model_name}/{self.path}/{self.source}"):
            os.makedirs(f"../models/{self.model_name}/{self.path}/{self.source}")

        model=self.Regressor(**params)
        model.fit(self.features,self.labels)
        with open(f"../models/{self.model_name}/{self.path}/{self.source}/{self.type}_bidding.pkl","wb") as f:
            pickle.dump(model,f)



#----------------------Hyperparameters----------------------
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
    'random_state':42,
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
    'random_state':42,
    'verbose':-1}


if __name__=="__main__":

    for source in ["dwd","gfs"]:
        
        for is_full in [True,False]:
    
            for target in ["solar","wind"]:
                
                configs={
                    "target_type":target,
                    "Regressor":LGBMRegressor,
                    "full":is_full, #Use the full amount of data
                    "model_name":"LGBM",
                    "source":source #NWP source
                    }
                trainer=Trainer(**configs)
                
                if target=="wind":
                    trainer.train(params_LGBM_wind_forecast)
                    trainer.train_bidding(params_LGBM_wind_trading)
                else:
                    trainer.train(params_LGBM_solar_forecast)
                    trainer.train_bidding(params_LGBM_solar_trading)
            
        
        
        
        
        
        
        
        
