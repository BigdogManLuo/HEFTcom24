import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import comp_utils

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle as pkl
from  tqdm import tqdm


#%% Weather Data
dwd_Hornsea1 = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_features=dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")


dwd_solar = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_features = dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")


#%% Energy Data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]



#%% Merge Energy and Weather Data
modelling_table = dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
modelling_table = modelling_table.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
modelling_table = modelling_table.drop(columns="ref_datetime",axis=1).reset_index()
modelling_table = modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")
modelling_table = modelling_table[modelling_table["valid_datetime"] - modelling_table["ref_datetime"] < np.timedelta64(50,"h")]
modelling_table.rename(columns={"WindSpeed:100":"WindSpeed"},inplace=True)


#%%
plt.figure(figsize=(9,5))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
sns.scatterplot(data=modelling_table, x="WindSpeed", y="Wind_MWh_credit",
                color='blue',s=5)
plt.xlabel('Wind Speed [m/s]')
plt.ylabel('Generation [MWh]')
plt.figure(figsize=(9,5))
sns.scatterplot(data=modelling_table, x="SolarDownwardRadiation", 
                y="Solar_MWh_credit", color='darkorange',s=5)
plt.xlabel('Solar Radiation Downwards [w/m^2]')
plt.ylabel('Generation [MWh]')


#%% Modelling
modelling_table = modelling_table[modelling_table["SolarDownwardRadiation"].notnull()]
modelling_table = modelling_table[modelling_table["WindSpeed"].notnull()]
modelling_table["total_generation_MWh"] = modelling_table["Wind_MWh_credit"] + modelling_table["Solar_MWh_credit"]

#只取modelling_table的前10%的数据
modelling_table = modelling_table.sort_values("valid_datetime").head(int(len(modelling_table)*0.2))


mod = smf.quantreg('total_generation_MWh ~ bs(SolarDownwardRadiation,df=5) + bs(WindSpeed,df=8)',
                   data=modelling_table)

forecast_models = dict()
for quantile in range(10,100,10):
    forecast_models[f"q{quantile}"] = mod.fit(q=quantile/100,max_iter=2500)
    modelling_table[f"q{quantile}"] = forecast_models[f"q{quantile}"].predict(modelling_table)
    modelling_table.loc[modelling_table[f"q{quantile}"] < 0, f"q{quantile}"] = 0
  
#%% Save  Model
for quantile in range(10,100,10):
    forecast_models[f"q{quantile}"].save(f"models/model_q{quantile}.pickle")

#%% Test

ref_time = modelling_table["ref_datetime"] == modelling_table["ref_datetime"][10]

plt.figure(figsize=(10,6))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
ax1 = sns.lineplot(data=modelling_table[ref_time], x="valid_datetime",
                   y="total_generation_MWh")

for quantile in range(10,100,10):
    sns.lineplot(data=modelling_table,
                 x=modelling_table[ref_time]["valid_datetime"],
                 y=modelling_table[ref_time][f"q{quantile}"],
                 color='gray',
                 alpha=1-abs(50-quantile)/50,
                 label=f'q{quantile}')

plt.ylim(0, 1600)
plt.xlim(modelling_table[ref_time]['valid_datetime'].min())
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M \n %b-%Y"))
plt.xlabel('Date/Time [30-minute period]')
plt.ylabel('Generation [MWh]')
plt.title(f"Forecast reference time: {modelling_table[ref_time]['ref_datetime'][0]}",
          fontsize=14)
plt.tight_layout()



#%% Forecast scoring
def pinball(y,q,alpha):
    return (y-q)*alpha*(y>=q) + (q-y)*(1-alpha)*(y<q)

def pinball_score(df):
    score = list()
    for qu in range(10,100,10):
        score.append(pinball(y=df["total_generation_MWh"],
            q=df[f"q{qu}"],
            alpha=qu/100).mean())
    return sum(score)/len(score)

print("pinball_score"+str(pinball_score(modelling_table)))



#%% Revenue scoring 

def revenue(bid,y,day_ahead_price,single_system_price):
    return bid*day_ahead_price + (y-bid)*(single_system_price - 0.07*(y-bid))

submission_date = "2020-09-21"
submission_date_forecast = modelling_table.loc[modelling_table["ref_datetime"]==pd.to_datetime(submission_date,utc=True)]
market_day = pd.DataFrame({"datetime":comp_utils.day_ahead_market_times(today_date=pd.to_datetime(submission_date))})
market_day = market_day.merge(submission_date_forecast,how="left",left_on="datetime",right_on="valid_datetime")
market_day["market_bid"] = market_day["q50"]

r=revenue(bid=market_day["market_bid"],
        y=market_day["total_generation_MWh"],
        day_ahead_price=market_day["DA_Price"],
        single_system_price=market_day["SS_Price"]).sum()


#%% Gnerate a competition submission

rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())


latest_dwd_Hornsea1 = comp_utils.weather_df_to_xr(rebase_api_client.get_hornsea_dwd())
latest_dwd_solar = comp_utils.weather_df_to_xr(rebase_api_client.get_pes10_nwp("DWD_ICON-EU"))

latest_dwd_Hornsea1_features = latest_dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
latest_dwd_solar_features = latest_dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()

latest_forecast_table = latest_dwd_Hornsea1_features.merge(latest_dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
latest_forecast_table = latest_forecast_table.set_index("valid_datetime").resample("30T").interpolate("linear",limit=5).reset_index()

latest_forecast_table.rename(columns={"WindSpeed:100":"WindSpeed"},inplace=True)


#%% Forecast 
for quantile in range(10,100,10):
    loaded_model = load_pickle(f"models/model_q{quantile}.pickle")
    latest_forecast_table[f"q{quantile}"] = loaded_model.predict(latest_forecast_table)
    
    # latest_forecast_table[f"q{quantile}"] = forecast_models[f"q{quantile}"].predict(latest_forecast_table)
    # latest_forecast_table.loc[latest_forecast_table[f"q{quantile}"] < 0, f"q{quantile}"] = 0

#%% Visualization
plt.figure(figsize=(10,6))
sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
ax1 = sns.lineplot(data=latest_forecast_table, x="valid_datetime",
                   y="q50")

for quantile in range(10,100,10):
    sns.lineplot(data=latest_forecast_table,
                 x=latest_forecast_table["valid_datetime"],
                 y=latest_forecast_table[f"q{quantile}"],
                 color='gray',
                 alpha=1-abs(50-quantile)/50,
                 label=f'q{quantile}')
plt.title(f"Forecast reference time: {latest_forecast_table['ref_datetime'][0]}"
          )
plt.xlabel('Date/Time [30-minute period]')
plt.ylabel('Generation [MWh]')
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d \n %Y"))
plt.ylim(0, 1600)
plt.xlim(latest_forecast_table['valid_datetime'].min())
plt.tight_layout()
plt.show()



#%% Submission
submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})
submission_data = submission_data.merge(latest_forecast_table,how="left",left_on="datetime",right_on="valid_datetime")
submission_data["market_bid"] = submission_data["q50"]

submission_data = comp_utils.prep_submission_in_json_format(submission_data)
print(submission_data)
#%%
rebase_api_client.submit(submission_data)


#%% Usage of other API Endpoints
test_date = "2023-11-08"

day_ahead_price=rebase_api_client.get_variable(day=test_date,variable="day_ahead_price")#.head()

#%% get submissions
submissions=rebase_api_client.get_submissions()

wind_production=rebase_api_client.get_wind_total_production()


