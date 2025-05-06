import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor

#%% Paparing data for correlation analysis
   
IntegratedDataset_dwd = pd.read_csv("../data/dataset/full/dwd/IntegratedDataset.csv")    
IntegratedDataset_gfs = pd.read_csv("../data/dataset/full/gfs/IntegratedDataset.csv")    

#Merge Datasets
IntegratedDataset=IntegratedDataset_gfs.merge(IntegratedDataset_dwd,how="inner",on=["ref_datetime","valid_datetime"])
IntegratedDataset.rename(columns={"Wind_MWh_credit_x":"Wind_MWh_credit","Solar_MWh_credit_x":"Solar_MWh_credit","total_generation_MWh_x":"total_generation_MWh","hours_x":"hours","DA_Price_x":"DA_Price","SS_Price_x":"SS_Price"},inplace=True)

#variables
Y_wind=IntegratedDataset["Wind_MWh_credit"]
Y_solar=IntegratedDataset["Solar_MWh_credit"]

#covariates
features_name_Y_wind=pd.read_csv("../data/dataset/train/dwd/WindDataset.csv").columns[:-1].tolist()
features_name_Y_solar=pd.read_csv("../data/dataset/train/dwd/SolarDataset.csv").columns[:-1].tolist()
features_name_Z=features_name_Y_wind+features_name_Y_solar

Z=IntegratedDataset[features_name_Z]

#%% Training the regression model (LightGBM)

#Wind
params_wind = {
    'objective': 'mse',
    'n_estimators': 1000,
    'num_leaves': 1000,
    'max_depth': 6,
    'learning_rate': 0.08,
    'min_data_in_leaf': 700,
    'lambda_l1': 70,
    'lambda_l2': 40,
    'verbose': -1
}

model_wind=LGBMRegressor(**params_wind)
model_wind.fit(Z,Y_wind)

#Solar
params_solar = {
    'objective': 'mse',
    'n_estimators': 2000,
    'num_leaves': 700,
    'max_depth': 9,
    'learning_rate': 0.063,
    'min_data_in_leaf': 1400,
    'lambda_l1': 80,
    'lambda_l2': 40,
    'verbose': -1
}

model_solar=LGBMRegressor(**params_solar)
model_solar.fit(Z,Y_solar)


#%% Test the independency of residuals

#Calculate residuals
Y_wind_hat=model_wind.predict(Z)
Y_solar_hat=model_solar.predict(Z)

residuals_wind=Y_wind-Y_wind_hat
residuals_solar=Y_solar-Y_solar_hat

residuals_wind = residuals_wind.values
residuals_solar = residuals_solar.values


#Mutual Information
mi = mutual_info_regression(residuals_solar.reshape(-1, 1), residuals_wind)

print("Mutual Information:",mi)

#%% Visualize the residuals

plt.rcParams['font.family'] = 'Times New Roman'

data = pd.DataFrame({'Residuals Wind': residuals_wind, 'Residuals Solar': residuals_solar})

sns.set(style='whitegrid')
g = sns.jointplot(x='Residuals Wind', y='Residuals Solar', data=data, kind='hist', bins=25, marginal_kws=dict(bins=25, fill=True, color='skyblue', alpha=0.8,edgecolor='black'),cmap='Blues')
g.set_axis_labels('Residuals of Wind Power (MWh)', 'Residuals of Solar Power (MWh)', fontsize=16,family='Times New Roman')
plt.xticks(fontsize=16,family='Times New Roman')

max_value=400
plt.xlim(-max_value,max_value)
plt.ylim(-max_value,max_value)

plt.tight_layout()
plt.savefig('../figs/correlation.png',dpi=800)


