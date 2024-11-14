import numpy as np
import pickle
from sklearn.linear_model import Lasso
from comp_utils import getLatestSolarGeneration,getSolarDatasetfromNC

#%% Create latest solar dataset
SolarGeneration=getLatestSolarGeneration()
modelling_table_solar=getSolarDatasetfromNC(SolarGeneration)

#%% Predict by Original Models
    
features_solar=modelling_table_solar.iloc[:,:-1]
labels_solar=modelling_table_solar.iloc[:,-1]

model_bidding=pickle.load(open("models/LGBM/full/dwd/solar_bidding.pkl","rb"))

output_solar=model_bidding.predict(features_solar)
output_solar[output_solar<1e-2]=0
output_solar[modelling_table_solar["hours"]<=3]=0
output_solar[modelling_table_solar["hours"]>=21]=0
modelling_table_solar["Predicted"]=output_solar

modelling_table_solar["Predicted^2"]=modelling_table_solar["Predicted"]**2

#%% Post Processing Models: Training and Testing

np.random.seed(65)
idxs_train=np.random.choice(modelling_table_solar.index,size=int(0.6*len(modelling_table_solar)),replace=False)
idxs_test=np.array(list(set(modelling_table_solar.index)-set(idxs_train)))
modelling_table_solar_train=modelling_table_solar.loc[idxs_train]
modelling_table_solar_test=modelling_table_solar.loc[idxs_test]

y=modelling_table_solar_test["Solar_MWh_credit"].values
y_hat=modelling_table_solar_test["Predicted"].values
Loss_MSE=np.mean((y-y_hat)**2)
print(f"Loss_MSE(Original): {Loss_MSE}")

params={
    "alpha":0.2
    }
model_revised=Lasso(**params)

features_train=modelling_table_solar_train[["Predicted","Predicted^2"]].values
labels_train=modelling_table_solar_train["Solar_MWh_credit"].values
model_revised.fit(features_train,labels_train)
output_revised_train=model_revised.predict(features_train)

features_test=modelling_table_solar_test[["Predicted","Predicted^2"]].values
labels_test=modelling_table_solar_test["Solar_MWh_credit"].values
output_revised=model_revised.predict(features_test)
output_revised[output_revised<1e-2]=0
output_revised[modelling_table_solar_test["hours"]<=3]=0
output_revised[modelling_table_solar_test["hours"]>=21]=0

Loss_mse_test=np.mean((labels_test-output_revised)**2)
print(f"Loss_MSE(Revised): {Loss_mse_test}")

print(f"coef——{model_revised.coef_} ")


#%% Full Model Training
params={
    "alpha":0.5
}
model_revised=Lasso(**params)

features=modelling_table_solar[["Predicted","Predicted^2"]].values
labels=modelling_table_solar["Solar_MWh_credit"].values

model_revised.fit(features,labels)

with open("models/Revised/solar_bidding.pkl","wb") as f:
    pickle.dump(model_revised,f)

print(f"coef——{model_revised.coef_} ")