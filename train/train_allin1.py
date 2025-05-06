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
            'num_leaves': 1000,
            'n_estimators': 2000,
            'max_depth':8,
            'min_data_in_leaf': 400,
            'learning_rate':0.2,
            'lambda_l1': 40,           
            'lambda_l2': 80,
            'verbose':-1
            }        
        Params_wind_forecast[f"q{quantile}"]=params

        params={
            'objective':'quantile',
            'alpha':quantile/100,
            'num_leaves': 1000,
            'n_estimators': 2000,
            'max_depth':14,
            'min_data_in_leaf': 2000,
            'learning_rate':0.19,
            'lambda_l1': 10,           
            'lambda_l2': 10,
            'verbose':-1
            }
        Params_solar_forecast[f"q{quantile}"]=params

    return Params_wind_forecast,Params_solar_forecast

class Trainer():

    def __init__(self,target_type,Regressor,full,model_name):

        self.type=target_type
        self.Regressor=Regressor
        self.full=full
        self.model_name=model_name
        if full==True:
            self.path="full"
        else:
            self.path="train"
        
        #Load dataset
        self.dataset=pd.read_csv(f"../data/dataset/{self.path}/allin1/{self.type.capitalize()}Dataset.csv")
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

        if not os.path.exists(f"../models/{self.model_name}/{self.path}/allin1"):
            os.makedirs(f"../models/{self.model_name}/{self.path}/allin1")

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            
            self.Models[f"q{quantile}"]=self.Regressor(**Params[f"q{quantile}"])
            self.Models[f"q{quantile}"].fit(self.features,self.labels)
            with open(f"../models/{self.model_name}/{self.path}/allin1/{self.type}_q{quantile}.pkl","wb") as f:
                pickle.dump(self.Models[f"q{quantile}"],f)


#----------------------Hyperparameters----------------------
params_LGBM_wind_forecast, params_LGBM_solar_forecast=getBestParams()

if __name__=="__main__":

        
    for is_full in [True,False]:

        for target in ["wind","solar"]:
            configs={
                "target_type":target,
                "Regressor":LGBMRegressor,
                "full":is_full, #Use the full amount of data
                "model_name":"LGBM",
                }
            trainer=Trainer(**configs)
            
            if target=="wind":
                trainer.train(params_LGBM_wind_forecast)
            else:
                trainer.train(params_LGBM_solar_forecast)
            
        
        
        
        
