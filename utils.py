import pickle
import pandas as pd
import numpy as np
import os
from statsmodels.iolib import smpickle
from tqdm import tqdm
import comp_utils
from itertools import chain
from scipy.interpolate import interp1d

def loadFeaturesandLabels(pathtype,source):

    WindDataset=pd.read_csv(f"data/dataset/{pathtype}/{source}/WindDataset.csv")
    SolarDataset=pd.read_csv(f"data/dataset/{pathtype}/{source}/SolarDataset.csv")

    features_wind=WindDataset.iloc[:,:-1].values
    labels_wind=WindDataset.iloc[:,-1].values

    features_solar=SolarDataset.iloc[:,:-1].values
    labels_solar=SolarDataset.iloc[:,-1].values

    return features_wind,labels_wind,features_solar,labels_solar

#============================Evaluation=================================
def pinball(y,y_hat,alpha):

    return ((y-y_hat)*alpha*(y>=y_hat) + (y_hat-y)*(1-alpha)*(y<y_hat)).mean()

def getRevenue(Xb,Xa,yd,ys):
    return yd*Xb+(Xa-Xb)*ys-0.07*(Xa-Xb)**2


def meanPinballLoss(y_true, y_pred):
    mpd=0
    for i in range(10,100,10):
        mpd+=pinball(y_true,y_pred[f"q{i}"],alpha=i/100)
    return mpd/9


#==========================Trainer=================================
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
        
        self.dataset=pd.read_csv(f"data/dataset/{self.path}/{source}/{self.type.capitalize()}Dataset.csv")
        self.features=self.dataset.iloc[:,:-1]
        self.labels=self.dataset.iloc[:,-1]

        if self.type=="wind":
            self.features=self.features
            self.labels=self.labels
        
        elif self.type=="solar":
            self.features=self.features[int(0.2*len(self.features)):]
            self.labels=self.labels[int(0.2*len(self.labels)):] 
        
        self.Models={}
        
    def train(self,Params):

        if not os.path.exists(f"models/{self.model_name}/{self.path}/{self.source}"):
            os.makedirs(f"models/{self.model_name}/{self.path}/{self.source}")

        for quantile in chain([0.1],tqdm(range(1,100,1)),[99.9]):
            
            self.Models[f"q{quantile}"]=self.Regressor(**Params[f"q{quantile}"])
            self.Models[f"q{quantile}"].fit(self.features,self.labels)
            with open(f"models/{self.model_name}/{self.path}/{self.source}/{self.type}_q{quantile}.pkl","wb") as f:
                pickle.dump(self.Models[f"q{quantile}"],f)

    def train_bidding(self,params):
        if not os.path.exists(f"models/{self.model_name}/{self.path}/{self.source}"):
            os.makedirs(f"models/{self.model_name}/{self.path}/{self.source}")

        model=self.Regressor(**params)
        model.fit(self.features,self.labels)
        with open(f"models/{self.model_name}/{self.path}/{self.source}/{self.type}_bidding.pkl","wb") as f:
            pickle.dump(model,f)


#==========================Forecasting=================================
def quantile_sort(forecast):

    quantiles = forecast.keys() #sorted(forecast.keys(), key=lambda x: int(x[1:]))
    forecast_array = np.array([forecast[q] for q in quantiles])
    forecast_array_sorted = np.sort(forecast_array, axis=0)
    for i, q in enumerate(quantiles):
        forecast[q] = forecast_array_sorted[i, :]
    
    return forecast

def forecast_wind(wind_features,full,WLimit,availableCapacities=None,model_name="LGBM",source="dwd"):

    if full:
        path="full"
    else:
        path="train"
 
    Wind_Generation_Forecast={}
    coff_ac=np.linspace(0.92,1,101)
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):

        with open(f"models/{model_name}/{path}/{source}/wind_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)

        output=model.predict(wind_features)
        output[output<0]=0

        #Plan Outage
        if WLimit:
            output=np.where(output>coff_ac[idx]*availableCapacities,coff_ac[idx]*availableCapacities,output)

        Wind_Generation_Forecast[f"q{quantile}"]=output
        
    #Quantile Sort
    Wind_Generation_Forecast=quantile_sort(Wind_Generation_Forecast)

    return Wind_Generation_Forecast

def forecast_wind_ensemble(wind_features_dwd,wind_features_gfs,full,WLimit,model_name="LGBM",availableCapacities=None):
    
    if full:
        path="full"
    else:
        path="train"
    
    #Separate Forecasting for dwd and gfs
    temp_dict={}
    for source in ["dwd","gfs"]:
        configs={
            "wind_features":wind_features_dwd if source=="dwd" else wind_features_gfs,
            "full":full,
            "source":source,
            "WLimit":False  # Ensemble first and then truncated
        }
        Wind_Generation_Forecast=forecast_wind(**configs)
        
        for quantile in chain([0.1],range(1,100,1),[99.9]):
            temp_dict[f"{source}_wind_q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]
    modelling_table=pd.DataFrame(temp_dict)
    
    #Ensemble Forecasting
    Wind_Generation_Forecast={}
    coff_ac=np.linspace(0.92,1,101)
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):
        with open(f"models/Ensemble/{path}/wind_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        feature_meta=modelling_table[[f"dwd_wind_q{quantile}",f"gfs_wind_q{quantile}"]].values
        #Forecast
        output=model.predict(feature_meta)
        output[output<0]=0
        
        #Plan Outage    
        if WLimit:
            output=np.where(output>coff_ac[idx]*availableCapacities,coff_ac[idx]*availableCapacities,output)
        
        Wind_Generation_Forecast[f"q{quantile}"]=output
    
    #Quantile Sort
    Wind_Generation_Forecast=quantile_sort(Wind_Generation_Forecast)

    return Wind_Generation_Forecast


def forecast_solar(solar_features,hours,full,SolarRevise,rolling_test=False,source="dwd",model_name="LGBM"):
    
    if full:
        path="full"
    else:
        path="train"

    Solar_Generation_Forecast={}
    for idx,quantile in enumerate(chain([0.1],range(1,100,1),[99.9])):

        #Load models
        with open(f"models/{model_name}/{path}/{source}/solar_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        
        #Forecasting
        output=model.predict(solar_features)
        output[output<1e-2]=0

        #Power at night is set to 0
        output[hours<=3]=0
        output[hours>=21]=0
        
        #Online Post-Processing
        if SolarRevise:
            if rolling_test:
                raise NotImplementedError("Rolling Test is not implemented yet")
            
            # For Competition use
            else:
                with open(f"models/Revised/solar_q{quantile}.pkl","rb") as f:
                    model_solar_revised=pickle.load(f)
                output_solar_origin=output.reshape(-1,1).copy()
                features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
                output=model_solar_revised.predict(features_revised)
            
            output[output<1e-2]=0

            #Power at night is set to 0
            output[hours<=3]=0
            output[hours>=21]=0
        
        Solar_Generation_Forecast[f"q{quantile}"]=output
    
    #Quantile Sort
    Solar_Generation_Forecast=quantile_sort(Solar_Generation_Forecast)

    return Solar_Generation_Forecast

def forecast_total(wind_features_dwd,wind_features_gfs,solar_features,hours,full,WLimit,SolarRevise,availableCapacities=None,rolling_test=False,dx=1e-1,aggregation="True"):
    
    params_wind={
        "wind_features_dwd":wind_features_dwd,
        "wind_features_gfs":wind_features_gfs,
        "full":full,
        "WLimit":WLimit,
        "availableCapacities":availableCapacities
    }

    params_solar={
        "solar_features":solar_features,
        "hours":hours,
        "full":full,
        "SolarRevise":SolarRevise,
        "rolling_test":rolling_test
    }

    Wind_Generation_Forecast=forecast_wind_ensemble(**params_wind)
    Solar_Generation_Forecast=forecast_solar(**params_solar)

    if aggregation:
        Total_generation_forecast=QuantileAdd(Wind_Generation_Forecast,Solar_Generation_Forecast,dx)
    else:
        Total_generation_forecast={}
        for quantile in range(10,100,10):
            Total_generation_forecast[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]+Solar_Generation_Forecast[f"q{quantile}"]

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast

def QuantileAdd(quantile_values_X,quantile_values_Y,dx):
    quantiles = quantile_values_X.keys()
    forecast_array_X = np.array([quantile_values_X[q] for q in quantiles])
    forecast_array_Y = np.array([quantile_values_Y[q] for q in quantiles])
    Total_generation_forecast={f"q{quantile}": np.zeros(forecast_array_X.shape[1]) for quantile in range(10,100,10)}
    quantiles = np.arange(0,101,1)
    for i in range(forecast_array_X.shape[1]):
        
        X_values=forecast_array_X[:,i]
        Y_values=forecast_array_Y[:,i]
        
        #If the PV intervals are small then just add them up
        if Y_values[90]>600:
            F_x,f_X,x_values=getPDF(quantiles,X_values,dx)
            F_y,f_Y,y_values=getPDF(quantiles,Y_values,dx)
             
            f_Z = np.convolve(f_X, f_Y, mode='full') * dx

            # Determine the new domain for Z
            z_min = x_values[0] + y_values[0]
            z_max = x_values[-1] + y_values[-1]
            z_range_0 = np.arange(z_min, z_max, dx)
            z_range_1 = np.arange(z_min, z_max+dx, dx)
            
            # Make sure f_Z is the same length as z_range_1 and z_range_0.
            if len(f_Z)!=len(z_range_0) and len(f_Z)!=len(z_range_1):
                print("Error")
                
            try:
                f_Z /= np.trapz(f_Z, z_range_0)
                z_range=z_range_0
            except:
                f_Z /= np.trapz(f_Z, z_range_1)
                z_range=z_range_1

            F_Z = np.cumsum(f_Z) * dx
            
            #Extract target quantiles value
            for quantile in range(10,100,10):
                Total_generation_forecast[f"q{quantile}"][i]=np.interp(quantile/100, F_Z, z_range)
        
        else:
            
            for quantile in range(10,100,10):
                Total_generation_forecast[f"q{quantile}"][i]=quantile_values_X[f"q{quantile}"][i]+quantile_values_Y[f"q{quantile}"][i]
    
    return Total_generation_forecast

def getPDF(quantiles,quantile_values,dx):

    probabilities = quantiles / 100.0
    cdf = interp1d(quantile_values, probabilities, kind='linear') 
    x_values = np.arange(quantile_values.min(), quantile_values.max(), dx)
    cdf_values = cdf(x_values)
    pdf_values = np.gradient(cdf_values, x_values)
    pdf_values = pdf_values / np.trapz(pdf_values, x_values)
    
    return cdf_values,pdf_values,x_values


#==========================Bidding================================

def forecast_bidding_wind(wind_features,full,source,WLimit,model_name="LGBM",availableCapacity=None):

    #Load models
    if full:
        path="full"
    else:
        path="train"
    with open(f"models/{model_name}/{path}/{source}/wind_bidding.pkl","rb") as f:
        model=pickle.load(f)

    #Forecast
    output=model.predict(wind_features)
    output[output<0]=0
    
    #Plan Outage
    if WLimit:
        output=np.where(output>0.95*availableCapacity,0.95*availableCapacity,output)
        
    return output

def forecast_bidding_wind_ensemble(wind_features_dwd,wind_features_gfs,full,WLimit,model_name="LGBM",availableCapacity=None):
        
    if full:
        path="full"
    else:
        path="train"
    
    temp_dict={}
    for source in ["dwd","gfs"]:
        configs={
            "wind_features":wind_features_dwd if source=="dwd" else wind_features_gfs,
            "full":full,
            "source":source,
            "WLimit":WLimit,
            "availableCapacity":availableCapacity
        }
        temp_dict[f"wind_bidding_{source}"]=forecast_bidding_wind(**configs)
    modelling_table=pd.DataFrame(temp_dict)
        
    '''
    with open(f"models/Ensemble/{path}/wind_bidding.pkl","rb") as f:
        model=pickle.load(f)
    '''
    feature_meta=modelling_table[["wind_bidding_dwd","wind_bidding_gfs"]]

    output=feature_meta.iloc[:,0]*0.58+feature_meta.iloc[:,1]*0.42
    output[output<0]=0
    
    return output

def forecast_bidding_solar(solar_features,hours,full,SolarRevise,rolling_test=False,source="dwd",model_name="LGBM"):
    
    if full:
        path="full"
    else:
        path="train"

    with open(f"models/{model_name}/{path}/{source}/solar_bidding.pkl","rb") as f:
        model=pickle.load(f)
    output=model.predict(solar_features)
    output[output<1e-2]=0
    output[hours<=3]=0
    output[hours>=21]=0

    if SolarRevise:
        if rolling_test:
            raise NotImplementedError("Rolling Test is not implemented yet")
        # For Competition use
        else:
            with open(f"models/Revised/solar_bidding.pkl","rb") as f:
                model_solar_revised=pickle.load(f)
            output_solar_origin=output.reshape(-1,1).copy()
            features_revised=np.concatenate([output_solar_origin,output_solar_origin**2],axis=1)
            output=model_solar_revised.predict(features_revised)

        output[output<1e-2]=0

        output[hours<=3]=0
        output[hours>=21]=0

    return output


def forecast_bidding(wind_features_dwd,wind_features_gfs,solar_features,full,hours,WLimit,SolarRevise,availableCapacity=None,rolling_test=False):

    params_wind={
        "wind_features_dwd":wind_features_dwd,
        "wind_features_gfs":wind_features_gfs,
        "full":full,
        "WLimit":WLimit,
        "availableCapacity":availableCapacity
    }

    params_solar={
        "solar_features":solar_features,
        "hours":hours,
        "full":full,
        "SolarRevise":SolarRevise,
        "rolling_test":rolling_test
    }

    wind_bidding=forecast_bidding_wind_ensemble(**params_wind)
    solar_bidding=forecast_bidding_solar(**params_solar)
    total_bidding=wind_bidding+solar_bidding

    return total_bidding




def forecastByBenchmark(wind_forecast_table,solar_forecat_table):

    #--------------------------------Load Models---------------------------
    Models_benchmark_wind={}
    Models_benchmark_solar={}
    
    for quantile in range(10,100,10):
        with open(f"models/benchmark/wind_q{quantile}.pickle", 'rb') as f:
            mod_wind = smpickle.load_pickle(f)
        with open(f"models/benchmark/solar_q{quantile}.pickle", 'rb') as f:
            mod_solar = smpickle.load_pickle(f)
        Models_benchmark_wind[f"q{quantile}"]=mod_wind
        Models_benchmark_solar[f"q{quantile}"]=mod_solar

    #--------------------------------Predict--------------------------------
    Total_generation_forecast={}
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    for quantile in range(10,100,10):
        
        #Forward
        output_wind=Models_benchmark_wind[f"q{quantile}"].predict(wind_forecast_table)
        output_solar=Models_benchmark_solar[f"q{quantile}"].predict(solar_forecat_table)
        
        #Post-Processing
        output_wind[output_wind<0]=0
        output_solar[output_solar<1e-2]=0
        
        total_generation_forecast=output_wind+output_solar
        #total_generation_forecast=[elem.item() for elem in total_generation_forecast]

        Total_generation_forecast[f"q{quantile}"]=total_generation_forecast
        Wind_Generation_Forecast[f"q{quantile}"]=output_wind
        Solar_Generation_Forecast[f"q{quantile}"]=output_solar

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast
