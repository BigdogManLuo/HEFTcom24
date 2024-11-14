import comp_utils
import pandas as pd
from comp_utils import send_mes
import matplotlib.pyplot as plt
import utils
import pickle
import numpy as np

'''-------------------------------Submit summary------------------------------'''
recordings="""
Weather source: dwd+gfs fusion
Forecast: LGBM
Bidding: MSE+historical average price difference correction
Wind power overall forecast
PV combined latest data+capacity correction
Total power generation forecast: quantile addition algorithm (quantile-CDF-PDF-convol-PDF-CDF-quantile)
"""

def initialSubmissionData():
    
    submission_data=pd.DataFrame({"datetime":comp_utils.day_ahead_market_times()})

    features_datetime=pd.concat([
        pd.DataFrame({"datetime":submission_data.iloc[0:1,0]-pd.Timedelta("30min")}),
        submission_data,
        pd.DataFrame({"datetime":submission_data.iloc[-1:,0]+pd.Timedelta("30min")})]
        ,axis=0).reset_index()
    
    return submission_data,features_datetime


def getFeatures(latest_Hornsea1,latest_solar,source):

    
    #===========================================Extract Features=================================================
    
    #Average wind speed
    latest_Hornsea1_features=latest_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()

    #Maximum wind speed
    latest_Hornsea1_features=latest_Hornsea1_features.merge(
        latest_Hornsea1["WindSpeed:100"].max(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"maxWindSpeed:100"}),
        how="left",on=["ref_datetime","valid_datetime"])

    #Minimum wind speed
    latest_Hornsea1_features=latest_Hornsea1_features.merge(
        latest_Hornsea1["WindSpeed:100"].min(dim=["latitude","longitude"]).to_dataframe().reset_index().rename(columns={"WindSpeed:100":"minWindSpeed:100"}),
        how="left",on=["ref_datetime","valid_datetime"]) 

    #Average solar radiation
    latest_solar_features=latest_solar[["SolarDownwardRadiation","CloudCover"]].mean(dim="point").to_dataframe().reset_index()
    
    #Maximum solar radiation
    latest_solar_features=latest_solar_features.merge(
        latest_solar[["SolarDownwardRadiation","CloudCover"]].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation","CloudCover":"maxCloudCover"}),
        how="left",on=["ref_datetime","valid_datetime"])
    
    #Minimum solar radiation
    latest_solar_features=latest_solar_features.merge(
        latest_solar[["SolarDownwardRadiation","CloudCover"]].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation","CloudCover":"minCloudCover"}),
        how="left",on=["ref_datetime","valid_datetime"])
    
    #===========================================Preprocessing=================================================

    #Merge Features
    latest_forecast_table = latest_Hornsea1_features.merge(latest_solar_features,how="outer",on=["ref_datetime","valid_datetime"])

    #Drop NA
    latest_forecast_table=latest_forecast_table.dropna()
    
    #Interpolate
    latest_forecast_table = latest_forecast_table.set_index("valid_datetime").resample("30T").interpolate("linear",limit=5).reset_index()

    #Prepare Submission Data
    submission_data,features_datetime=initialSubmissionData()

    #Extract latest_forecast_table according to features_datetime
    latest_forecast_table=latest_forecast_table[latest_forecast_table["valid_datetime"].isin(features_datetime["datetime"])].reset_index()

    #Create Integrated Features
    IntegratedFeatures=pd.DataFrame()
    for i in range(len(latest_forecast_table)-2):

        feature={
            
            f"ws_100_t-1_{source}_1":latest_forecast_table.iloc[i]["WindSpeed:100"],
            f"ws_100_t-1_{source}_max":latest_forecast_table.iloc[i]["maxWindSpeed:100"],
            f"ws_100_t-1_{source}_min":latest_forecast_table.iloc[i]["minWindSpeed:100"],
            
            f"ws_100_t_{source}_1":latest_forecast_table.iloc[i+1]["WindSpeed:100"],
            f"ws_100_t_{source}_max":latest_forecast_table.iloc[i+1]["maxWindSpeed:100"],
            f"ws_100_t_{source}_min":latest_forecast_table.iloc[i+1]["minWindSpeed:100"],
            
            f"ws_100_t+1_{source}_1":latest_forecast_table.iloc[i+2]["WindSpeed:100"],
            f"ws_100_t+1_{source}_max":latest_forecast_table.iloc[i+2]["maxWindSpeed:100"],
            f"ws_100_t+1_{source}_min":latest_forecast_table.iloc[i+2]["minWindSpeed:100"],

            f"rad_t-1_{source}":latest_forecast_table.iloc[i]["SolarDownwardRadiation"],
            f"rad_t-1_{source}_max":latest_forecast_table.iloc[i]["maxSolarDownwardRadiation"],
            f"rad_t-1_{source}_min":latest_forecast_table.iloc[i]["minSolarDownwardRadiation"],
            
            f"rad_t_{source}":latest_forecast_table.iloc[i+1]["SolarDownwardRadiation"],
            f"rad_t_{source}_max":latest_forecast_table.iloc[i+1]["maxSolarDownwardRadiation"],
            f"rad_t_{source}_min":latest_forecast_table.iloc[i+1]["minSolarDownwardRadiation"],

            f"rad_t+1_{source}":latest_forecast_table.iloc[i+2]["SolarDownwardRadiation"],
            f"rad_t+1_{source}_max":latest_forecast_table.iloc[i+2]["maxSolarDownwardRadiation"],
            f"rad_t+1_{source}_min":latest_forecast_table.iloc[i+2]["minSolarDownwardRadiation"],
            
            f"cloudcov_t-1_{source}":latest_forecast_table.iloc[i]["CloudCover"],
            f"cloudcov_t-1_{source}_max":latest_forecast_table.iloc[i]["maxCloudCover"],
            f"cloudcov_t-1_{source}_min":latest_forecast_table.iloc[i]["minCloudCover"],

            f"cloudcov_t_{source}":latest_forecast_table.iloc[i+1]["CloudCover"],
            f"cloudcov_t_{source}_max":latest_forecast_table.iloc[i+1]["maxCloudCover"],
            f"cloudcov_t_{source}_min":latest_forecast_table.iloc[i+1]["minCloudCover"],

            f"cloudcov_t+1_{source}":latest_forecast_table.iloc[i+2]["CloudCover"],
            f"cloudcov_t+1_{source}_max":latest_forecast_table.iloc[i+2]["maxCloudCover"],
            f"cloudcov_t+1_{source}_min":latest_forecast_table.iloc[i+2]["minCloudCover"],

            "ref_datetime":latest_forecast_table.iloc[i+1]["ref_datetime"],
            "valid_datetime":latest_forecast_table.iloc[i+1]["valid_datetime"],
        }
        IntegratedFeatures=IntegratedFeatures._append(feature,ignore_index=True)

    IntegratedFeatures["hours"]=pd.to_datetime(IntegratedFeatures["valid_datetime"]).dt.hour
    hours=IntegratedFeatures["hours"]

    columns_wind_features=pd.read_csv(f"data/dataset/full/{source}/WindDataset.csv").columns.tolist()[:-1]
    columns_solar_features=pd.read_csv(f"data/dataset/full/{source}/SolarDataset.csv").columns.tolist()[:-1]
    wind_features=IntegratedFeatures[columns_wind_features]
    solar_features=IntegratedFeatures[columns_solar_features]
    
    return IntegratedFeatures,wind_features,solar_features,hours,submission_data


#%% Data Preparation

# Read API Key
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())   

# Get latest weather data
latest_dwd_Hornsea1 = comp_utils.weather_df_to_xr(rebase_api_client.get_hornsea_dwd())
latest_dwd_solar = comp_utils.weather_df_to_xr(rebase_api_client.get_pes10_nwp("DWD_ICON-EU"))
latest_dwd_demand=comp_utils.weather_df_to_xr(rebase_api_client.get_demand_nwp("DWD_ICON-EU"))
latest_gfs_Hornsea1 = comp_utils.weather_df_to_xr(rebase_api_client.get_hornsea_gfs())
latest_gfs_solar=comp_utils.weather_df_to_xr(rebase_api_client.get_pes10_nwp("NCEP_GFS"))
latest_gfs_demand=comp_utils.weather_df_to_xr(rebase_api_client.get_demand_nwp("NCEP_GFS"))

#Extract Features
IntegratedFeatures_dwd,wind_features_dwd,solar_features_dwd,hours_dwd,submission_data=getFeatures(latest_dwd_Hornsea1,latest_dwd_solar,"dwd")
IntegratedFeatures_gfs,wind_features_gfs,solar_features_gfs,hours_gfs,_=getFeatures(latest_gfs_Hornsea1,latest_gfs_solar,"gfs")
    

#%% Pre-Forecast by Benchmark Model
wind_forecast_table=wind_features_dwd["ws_100_t_dwd_1"].to_frame()
wind_forecast_table.rename(columns={"ws_100_t_dwd_1":"WindSpeed"},inplace=True)
solar_forecat_table=solar_features_dwd["rad_t_dwd"].to_frame()
solar_forecat_table.rename(columns={"rad_t_dwd":"SolarDownwardRadiation"},inplace=True)

pre_Total_Generation_Forecast,pre_Wind_Generation_Forecast,pre_Solar_Generation_Forecast=utils.forecastByBenchmark(wind_forecast_table,solar_forecat_table)

#%% Forecast
params={
    "wind_features_dwd":wind_features_dwd,
    "wind_features_gfs":wind_features_gfs,
    "solar_features":solar_features_dwd,
    "hours":hours_dwd,
    "full":True,
    "WLimit":True,
    "SolarRevise":True,
    "availableCapacities":390,   #According to Remit message
    "aggregation":True
}
Total_Generation_Forecast,Wind_Generation_Forecast,Solar_Generation_Forecast=utils.forecast_total(**params)

for quantile in range(10,100,10):
    submission_data[f"q{quantile}"]=Total_Generation_Forecast[f"q{quantile}"]


#%% Bidding
status_bidding=rebase_api_client.freshPast30daysPrices()
pd_hourly_mean=comp_utils.getHourlyPriceDiff()


params={
    "wind_features_dwd":wind_features_dwd,
    "wind_features_gfs":wind_features_gfs,
    "solar_features":solar_features_dwd,
    "full":True,
    "hours":hours_dwd,
    "WLimit":True,
    "availableCapacity":390,
    "SolarRevise":True
}
biddings_0=utils.forecast_bidding(**params).values

biddings=biddings_0.copy()
for hour in range(24):
    biddings[hours_dwd==hour]+=7.14*pd_hourly_mean[hour]

biddings[biddings<0]=0
biddings[biddings>1800]=1800

submission_data["market_bid"]=biddings


#%% Logs
tomorrow=pd.Timestamp.now().date()+pd.Timedelta("1D")
IntegratedFeatures_dwd.to_csv(f"logs/dfs/{tomorrow}_IntegratedFeatures_dwd.csv",index=False)
IntegratedFeatures_gfs.to_csv(f"logs/dfs/{tomorrow}_IntegratedFeatures_gfs.csv",index=False)

#Record Weather Data
latest_dwd_Hornsea1.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_dwd_Hornsea1.csv",index=False)
latest_dwd_solar.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_dwd_solar.csv",index=False)
latest_dwd_demand.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_dwd_demand.csv",index=False)
latest_gfs_Hornsea1.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_gfs_Hornsea1.csv",index=False)
latest_gfs_solar.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_gfs_solar.csv",index=False)
latest_gfs_demand.to_dataframe().reset_index().to_csv(f"logs/weather/{tomorrow}_latest_gfs_demand.csv",index=False)


#Record Submission Data
submission_data.to_csv(f"logs/sub_data/{tomorrow}_submission_data.csv",index=False)


#Record Wind and Solar Forecast Results
with open(f"logs/forecast/{tomorrow}_Wind_Generation_Forecast.pkl","wb") as f:
    pickle.dump(Wind_Generation_Forecast,f)
with open(f"logs/forecast/{tomorrow}_Solar_Generation_Forecast.pkl","wb") as f:
    pickle.dump(Solar_Generation_Forecast,f)



#%%=====================================Visualization=====================================
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(Wind_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
plt.ylabel("Wind Generation")
plt.xlabel("Time")
plt.grid()
plt.subplot(3,1,2)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(Solar_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}")
plt.ylabel("Solar Generation")
plt.xlabel("Time")
plt.grid()
plt.subplot(3,1,3)
for idx,quantile in enumerate(range(10,100,10)):
    plt.plot(submission_data[f"q{quantile}"],label=f"q{quantile}")
plt.plot(biddings_0,label="biddings0",color="green",linestyle="--",linewidth=2) #Initial bidding result
plt.plot(submission_data["market_bid"],label="bidding",color="black",linestyle="--",linewidth=2) #Final bidding result
plt.ylabel("TotalGeneration")
plt.xlabel("Time")
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 2), loc='upper left',frameon=False)
plt.savefig(f"logs/figs/{tomorrow}_forecast.png",dpi=500)

# =====================================Compared with Benchmark=====================================
plt.figure(figsize=(8,6))
plt.subplot(3,1,1)
plt.plot(pre_Wind_Generation_Forecast["q50"],label="benchmark")
plt.plot(Wind_Generation_Forecast["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Wind Generation")
plt.grid()

plt.subplot(3,1,2)
plt.plot(pre_Solar_Generation_Forecast["q50"],label="benchmark")
plt.plot(Solar_Generation_Forecast["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Solar Generation")
plt.grid()

plt.subplot(3,1,3)
plt.plot(pre_Total_Generation_Forecast["q50"],label="benchmark")
plt.plot(submission_data["q50"],label="now")
plt.legend()
plt.ylabel("q50")
plt.xlabel("Total Generation")
plt.grid()
plt.tight_layout()
plt.savefig(f"logs/figs/{tomorrow}_benchmark_comp.png",dpi=500)
plt.legend()

#=============================Bidding Comparison================================
plt.figure(figsize=(8,6))
plt.plot(biddings_0,label="biddings0")
plt.plot(biddings,label="biddings")
plt.plot(Total_Generation_Forecast["q50"],label="q50")
plt.grid()
plt.legend()
plt.savefig(f"logs/figs/{tomorrow}_bidding_comp.png",dpi=500)


#%% Submission
submission_data = comp_utils.prep_submission_in_json_format(submission_data)

resp=rebase_api_client.submit(submission_data,recordings+"\n Electricity Price:"+status_bidding) 
submissions=rebase_api_client.get_submissions(market_day=tomorrow)

#Send Email Notification
send_mes(recordings+"\n Electricity Price:"+status_bidding,resp,tomorrow)
if resp.status_code==200:
   print("Submission Successful")
else:
    raise Exception("Submission Failed")
