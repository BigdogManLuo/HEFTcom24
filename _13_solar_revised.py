import numpy as np
import pickle
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
from comp_utils import pinball
from comp_utils import getLatestSolarGeneration,getSolarDatasetfromNC
from itertools import chain
import matplotlib.pyplot as plt


#%% Create latest solar dataset
SolarGeneration=getLatestSolarGeneration()
modelling_table_solar=getSolarDatasetfromNC(SolarGeneration,is_train=False)


#%% Predict by Original Models

features_solar=modelling_table_solar.iloc[:,:-1]
labels_solar=modelling_table_solar.iloc[:,-1]

Models_solar={}
for quantile in chain([0.1],range(1,100,1),[99.9]):
    with open(f"models/LGBM/full/dwd/solar_q{quantile}.pkl","rb") as f:
        Models_solar[f"q{quantile}"]=pickle.load(f)

for quantile in chain([0.1],range(1,100,1),[99.9]):
    output_solar=Models_solar[f"q{quantile}"].predict(features_solar)
    output_solar[output_solar<1e-2]=0
    output_solar[modelling_table_solar["hours"]<=3]=0
    output_solar[modelling_table_solar["hours"]>=21]=0
    modelling_table_solar[f"q{quantile}"]=output_solar

#Extend features
for quantile in chain([0.1],range(1,100,1),[99.9]):
    modelling_table_solar[f"q{quantile}^2"]=modelling_table_solar[f"q{quantile}"]**2


#%% Post Processing Models: Training and Testing

np.random.seed(10) #10 20 30 40 11.6 12.26 11.34 11.41
idxs_train=np.random.choice(modelling_table_solar.index,size=int(0.6*len(modelling_table_solar)),replace=False)
idxs_test=np.array(list(set(modelling_table_solar.index)-set(idxs_train)))
modelling_table_solar_train=modelling_table_solar.loc[idxs_train]
modelling_table_solar_test=modelling_table_solar.loc[idxs_test]

Pbloss=0
for quantile in chain([0.1],range(1,100,1),[99.9]):
    Pbloss+=pinball(y=modelling_table_solar_test["Solar_MWh_credit"].values,
                    y_hat=modelling_table_solar_test[f"q{quantile}"].values,
                    alpha=quantile/100)
Pbloss=Pbloss/101
print(f"pinball_score(Original): {Pbloss.mean()}")

Pbloss_test=[]
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
for quantile in chain([0.1],range(1,100,1),[99.9]):

    params={
        "quantile":quantile/100,
        "alpha":0,
        "solver":solver
    }
    model_revised=QuantileRegressor(**params)

    
    features_train=modelling_table_solar_train[[f"q{quantile}",f"q{quantile}^2"]].values
    labels_train=modelling_table_solar_train["Solar_MWh_credit"].values
    model_revised.fit(features_train,labels_train)

    features_test=modelling_table_solar_test[[f"q{quantile}",f"q{quantile}^2"]].values
    output_revised=model_revised.predict(features_test)
    output_revised[output_revised<1e-2]=0
    output_revised[modelling_table_solar_test["hours"]<=3]=0
    output_revised[modelling_table_solar_test["hours"]>=21]=0

    Pbloss_test.append(pinball(y=modelling_table_solar_test["Solar_MWh_credit"].values,y_hat=output_revised,alpha=quantile/100))

    print(f"coef——{quantile}:{model_revised.coef_} ")

Pbloss_test=np.array(list(Pbloss_test)).mean()
print(f"pinball_score(Revised): {Pbloss_test}")

                
#%% Full Model Training
solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
for quantile in range(10,100,10):
    
    params={
        "quantile":quantile/100,
        "alpha":0.1,
        "solver":solver
    }
    model_revised=QuantileRegressor(**params)

    features=modelling_table_solar[[f"q{quantile}",f"q{quantile}^2"]].values
    labels=modelling_table_solar["Solar_MWh_credit"].values
    model_revised.fit(features,labels)

    with open(f"models/Revised/solar_q{quantile}.pkl","wb") as f:
        pickle.dump(model_revised,f)

    print(f"coef——{quantile}:{model_revised.coef_} ")