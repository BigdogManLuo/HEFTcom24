import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#Load data
energy_data=pd.read_csv('data/raw/Energy_Data_20200920_20240118.csv')
energy_data=energy_data.dropna()
energy_data["hours"]=pd.to_datetime(energy_data["dtm"]).dt.hour

wind_power=0.5*energy_data['Wind_MW']-energy_data['boa_MWh']
pv_power=0.5*energy_data['Solar_MW']
pv_power=pv_power[(energy_data["hours"]>=7) & (energy_data["hours"]<=18)] #Retain only daylight hours
wind_power=wind_power[(energy_data["hours"]>=7) & (energy_data["hours"]<=18)]

#Mutual Information
mi = mutual_info_regression(wind_power.values.reshape(-1, 1), pv_power)

print("Mutual Information:",mi)


#Correlation
data = pd.DataFrame({'Wind Power': wind_power, 'PV Power': pv_power})
sns.set(style='whitegrid')
g = sns.jointplot(x='Wind Power', y='PV Power', data=data, kind='hist', bins=25, marginal_kws=dict(bins=25, fill=True, color='skyblue', alpha=0.8,edgecolor='black'),cmap='Blues')
g.set_axis_labels('Wind Power (MW)', 'Solar Power (MW)', fontsize=16,family='Times New Roman')
plt.xticks(fontsize=16,family='Times New Roman')
plt.yticks(fontsize=16,family='Times New Roman')
plt.tight_layout()
plt.savefig('figs/correlation.png',dpi=800)

