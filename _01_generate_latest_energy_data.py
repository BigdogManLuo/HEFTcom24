import pandas as pd
from tqdm import tqdm
import comp_utils

#获取2023-10-27之后的发电数据
rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())

#初始化一个空的dataframe
energy_data_latest=pd.DataFrame()

#遍历"2023-10-27"到"2024-01-08"之间的日期
for day in tqdm(pd.date_range("2024-01-19","2024-01-27")):
     
    #将day转换为带UTC时区的数据
    day=day.tz_localize("UTC")

    #每天所需时段
    #timestamp_utc=pd.DataFrame({"timestamp_utc": pd.date_range(start=(day - pd.Timedelta(days=1)).replace(hour=22, minute=0, second=0),end=day.replace(hour=21, minute=30, second=0),freq="30min")})

    try:
        #获取当天的风电数据
        wind_tmp=rebase_api_client.get_variable(day=day.strftime("%Y-%m-%d"),variable="wind_total_production")
        wind_tmp["timestamp_utc"]=pd.to_datetime(wind_tmp["timestamp_utc"])

        #只取每天所需时段
        #wind_tmp=timestamp_utc.merge(wind_tmp,how="left",on="timestamp_utc")

    except:
        #创建一个含nan的dataframe
        columns=["timestamp_utc","generation_mw","boa"]
        wind_tmp=pd.DataFrame(columns=columns)
        #wind_tmp["timestamp_utc"]=timestamp_utc["timestamp_utc"]

    try:
        #获取当天的光伏数据
        solar_tmp=rebase_api_client.get_variable(day=day.strftime("%Y-%m-%d"),variable="solar_total_production")
        solar_tmp["timestamp_utc"]=pd.to_datetime(solar_tmp["timestamp_utc"])

        #只取每天所需时段
        #solar_tmp=timestamp_utc.merge(solar_tmp,how="left",on="timestamp_utc")

    except:
        #创建一个含nan的dataframe
        columns=["timestamp_utc","generation_mw","installed_capacity_mwp","capacity_mwp"]
        solar_tmp=pd.DataFrame(columns=columns)
        #solar_tmp["timestamp_utc"]=timestamp_utc["timestamp_utc"]


    # 将当天的风电数据和光伏数据加入到energy_data_latest
    energy_data_today=pd.DataFrame({"dtm": pd.to_datetime(wind_tmp["timestamp_utc"]),
        "boa": wind_tmp["boa"],
        "Wind_MW": wind_tmp["generation_mw"],
        "Solar_MW": solar_tmp["generation_mw"],
        "installed_capacity_mwp": solar_tmp["installed_capacity_mwp"],
        "capacity_mwp": solar_tmp["capacity_mwp"]
        })
    
    energy_data_latest = pd.concat([energy_data_latest, energy_data_today], ignore_index=True)

#把2024-01-22 22：30之后的数据删除
energy_data_latest = energy_data_latest[energy_data_latest["dtm"] <= pd.Timestamp("2024-01-22 22:30", tz='UTC')]

#保存energy_data_latest
energy_data_latest.to_csv("data/Energy_Data_latest.csv",index=False)







