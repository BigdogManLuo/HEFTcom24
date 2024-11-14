from requests import Session
import requests
import pandas as pd
import datetime
import warnings
import smtplib
import matplotlib.cm as cm
import numpy as np
from matplotlib import pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.image import MIMEImage
from tqdm import tqdm
import xarray as xr

class RebaseAPI:
    
  challenge_id = 'heftcom2024'   
  base_url = 'https://api.rebase.energy'

  def __init__(
    self,
    api_key = open("team_key.txt").read()
    ):
    self.api_key = api_key
    self.headers = {
      'Authorization': f"Bearer {api_key}"
      }
    self.session = Session()
    self.session.headers = self.headers

  def get_variable(
      self,
      day: str,
      variable: ["market_index",
                 "day_ahead_price",
                 "imbalance_price",
                 "wind_total_production",
                 "solar_total_production",
                 "solar_and_wind_forecast"
                 ],
                 ):
    url = f"{self.base_url}/challenges/data/{variable}"
    params = {'day': day}
    resp = self.session.get(url, params=params)

    data = resp.json()
    df = pd.DataFrame(data)
    return df  


  # Solar and wind forecast
  def get_solar_wind_forecast(self,day):
    url = f"{self.base_url}/challenges/data/solar_and_wind_forecast"
    params = {'day': day}
    resp = self.session.get(url, params=params)
    data = resp.json()
    df = pd.DataFrame(data)
    return df


  # Day ahead demand forecast
  def get_day_ahead_demand_forecast(self):
    url = f"{self.base_url}/challenges/data/day_ahead_demand"
    resp = self.session.get(url)
    print(resp)
    return resp.json()


  # Margin forecast
  def get_margin_forecast(self):
    url = f"{self.base_url}/challenges/data/margin_forecast"
    resp = self.session.get(url)
    print(resp)
    return resp.json()


  def query_weather_latest(self,model, lats, lons, variables, query_type):
    url = f"{self.base_url}/weather/v2/query"

    body = {
        'model': model,
        'latitude': lats,
        'longitude': lons,
        'variables': variables,
        'type': query_type,
        'output-format': 'json',
        'forecast-horizon': 'latest'
    }

    resp = requests.post(url, json=body, headers={'Authorization': self.api_key})
    print(resp.status_code)

    return resp.json()


  def query_weather_latest_points(self,model, lats, lons, variables):
    # Data here is returned a list
    data = self.query_weather_latest(model, lats, lons, variables, 'points')

    df = pd.DataFrame()
    for point in range(len(data)):
      new_df = pd.DataFrame(data[point])
      new_df["point"] = point
      new_df["latitude"] = lats[point]
      new_df["longitude"] = lons[point]
      df = pd.concat([df,new_df])

    return df


  def query_weather_latest_grid(self,model, lats, lons, variables):
    # Data here is returned as a flattened
    data = self.query_weather_latest(model, lats, lons, variables, 'grid')
    df = pd.DataFrame(data)

    return df


  # To query Hornsea project 1 DWD_ICON-EU grid
  def get_hornsea_dwd(self):
    # As a 6x6 grid
    lats = [53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
    lons = [1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

    variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'
    return self.query_weather_latest_grid('DWD_ICON-EU', lats, lons, variables)


  # To query Hornsea project 1 GFS grid
  def get_hornsea_gfs(self):
    # As a 3x3 grid
    lats = [53.59, 53.84, 54.09]
    lons = [1.522, 1.772, 2.022]

    variables = 'WindSpeed, WindSpeed:100, WindDirection, WindDirection:100, Temperature, RelativeHumidity'
    return self.query_weather_latest_grid('NCEP_GFS', lats, lons, variables)


  def get_pes10_nwp(self,model):
    # As a list of points
    lats = [52.4872562, 52.8776682, 52.1354277, 52.4880497, 51.9563696, 52.2499177, 52.6416477, 52.2700912, 52.1960768, 52.7082618, 52.4043468, 52.0679429, 52.024023, 52.7681276, 51.8750506, 52.5582373, 52.4478922, 52.5214863, 52.8776682, 52.0780721]
    lons = [0.4012455, 0.7906532, -0.2640343, -0.1267052, 0.6588173, 1.3894081, 1.3509559, 0.7082557, 0.1534462, 0.7302284, 1.0762977, 1.1751747, 0.2962684, 0.1699257, 0.9115028, 0.7137489, 0.1204872, 1.5706825, 1.1916542, -0.0113488]

    variables = 'SolarDownwardRadiation, CloudCover, Temperature,RelativeHumidity,TotalPrecipitation,PressureReducedMSL'
    return self.query_weather_latest_points(model, lats, lons, variables)
  

  def get_demand_nwp(self,model):
    # As list of points
    lats = [51.479, 51.453, 52.449, 53.175, 55.86, 53.875, 54.297]
    lons = [-0.451, -2.6, -1.926, -2.986, -4.264, -0.442, -1.533]

    variables = 'Temperature, WindSpeed, WindDirection, TotalPrecipitation, RelativeHumidity'
    return self.query_weather_latest_points(model, lats, lons, variables)


  def submit(self,data,recordings):

    url = f"{self.base_url}/challenges/{self.challenge_id}/submit"

    resp = self.session.post(url,headers=self.headers, json=data)
    
    print(resp)
    print(resp.text)

    # Write log file
    text_file = open(f"logs/submissions/sub_{pd.Timestamp('today').strftime('%Y%m%d-%H%M%S')}.txt", "w")
    text_file.write(resp.text)
    
    #Log the submission
    text_file.write("\n")
    text_file.write("\n")
    text_file.write(recordings)

    text_file.close()

    return resp

  def get_submissions(self,market_day):
    url = f"{self.base_url}/challenges/{self.challenge_id}/submissions?market_day={market_day}"
    resp = self.session.get(url)
    print(resp)
    return resp.json()
  
  def get_history_score(self,day):

    submissions=self.get_submissions(market_day=day)
    solution=submissions["items"][-1]["solution"]
    print(f"market day: {solution['market_day']}")
    submission=solution['submission']

    Total_Generation_Forecast={f"q{quantile}":[] for quantile in range(10,100,10)}

    for quantile in range(10,100,10):
        for i in range(48):
            Total_Generation_Forecast[f"q{quantile}"].append(submission[i]["probabilistic_forecast"][f'{quantile}'])


    #=======================Get True Generation=======================
    wind_generation=self.get_variable(day=day,variable="wind_total_production")
    wind_generation["Wind_MWh_Credit"]=0.5*wind_generation["generation_mw"]-wind_generation["boa"]


    solar_generation=self.get_variable(day=day,variable="solar_total_production")
    solar_generation["Solar_MWh_Credit"]=0.5*solar_generation["generation_mw"]

    Total_Generation_True=wind_generation["Wind_MWh_Credit"]+solar_generation["Solar_MWh_Credit"]
    Total_Generation_True=Total_Generation_True[0:48]

    #=======================Get Pinball Score=======================
    pinball_score=0
    for quantile in range(10,100,10):
        pinball_score+=pinball(y=Total_Generation_True,y_hat=Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
    pinball_score=pinball_score/9
    print(f"pinball_score: {pinball_score}")

    #=======================Plot=======================
    cmap = cm.get_cmap('gray_r')
    colors = cmap(np.abs(np.linspace(-1, 1, 9)**2))
    plt.figure(figsize=(8,6))
    for quantile in range(10,100,10):
        plt.plot(Total_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}",color=colors[quantile//10-1])
        
    plt.plot(Total_Generation_True,label="True")
    plt.legend()
    plt.title(f"{solution['market_day']}")
    plt.show()

  def freshPast30daysPrices(self):

    history_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=61)
    current_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=4)
    DAP=self.get_variable(day=current_day.strftime("%Y-%m-%d"),variable="day_ahead_price")
    SSP=self.get_variable(day=current_day.strftime("%Y-%m-%d"),variable="imbalance_price")
    
    #Check if DAP and SSP are empty dataframes
    if len(DAP)==0 or len(SSP)==0:
        status="No latest data availab"
        print(status)

    #Check if DAP and SSP contain missing values
    elif DAP.isnull().values.any() or SSP.isnull().values.any():
        status="There are missing values in the data"
        print(status)

    #Rolling update of historical price data
    else:
      DAP.rename(columns={"price":"day_ahead_price"},inplace=True)
      DAP["imbalance_price"]=SSP["imbalance_price"]
      DAP["price_diff"]=DAP["day_ahead_price"]-DAP["imbalance_price"]
      DAP["timestamp_utc"]=pd.to_datetime(DAP["timestamp_utc"])
      DAP["hour"]=DAP["timestamp_utc"].dt.hour
      
      PD=pd.read_csv("data/rolling_past_price.csv")
      PD["timestamp_utc"] = pd.to_datetime(PD["timestamp_utc"])
      PD=PD[PD["timestamp_utc"]>history_day]
      PD=pd.concat([PD,DAP],axis=0)
      PD=PD.drop_duplicates(subset=["timestamp_utc"],keep="first")
      PD.to_csv("data/rolling_past_price.csv",index=False)
      status="Successfully updated"
      
    return status
    
def getHourlyPriceDiff():
    PD=pd.read_csv("data/rolling_past_price.csv")
    pd_hourly_mean=PD["price_diff"].groupby(PD["hour"]).mean()
    
    plt.figure()
    plt.plot(pd_hourly_mean,marker="s")
    plt.grid()
    plt.title("Hourly Price Difference")

    return pd_hourly_mean

def getHourlyPriceMean(price_type):
    price_data=pd.read_csv("data/rolling_past_price.csv")
    if price_type=="day_ahead":
        pd_hourly_mean=price_data["day_ahead_price"].groupby(price_data["hour"]).mean()
    elif price_type=="imbalance":
        pd_hourly_mean=price_data["imbalance_price"].groupby(price_data["hour"]).mean()
        
    plt.figure()
    plt.plot(pd_hourly_mean)
    plt.grid()

    return pd_hourly_mean


def pinball(y,y_hat,alpha):
    
    return ((y-y_hat)*alpha*(y>=y_hat) + (y_hat-y)*(1-alpha)*(y<y_hat)).mean()

def getRevenue(Xb,Xa,yd,ys):
    
    return yd*Xb+(Xa-Xb)*ys-0.07*(Xa-Xb)**2


def getLatestSolarGeneration():
    rebase_api_client = RebaseAPI(api_key = open("team_key.txt").read())
    current_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=1)
    SolarGeneration=pd.DataFrame()
    for day in tqdm(pd.date_range("2024-03-15",current_day.strftime("%Y-%m-%d"))):
        day=day.strftime("%Y-%m-%d")
        solar_generation=rebase_api_client.get_variable(day=day,variable="solar_total_production")
        solar_generation["Solar_MWh_credit"]=0.5*solar_generation["generation_mw"]
        solar_generation["timestamp_utc"]=pd.to_datetime(solar_generation["timestamp_utc"],utc=True)
        SolarGeneration=pd.concat([SolarGeneration,solar_generation],axis=0)
    return SolarGeneration

def getLatestWindGeneration():
    rebase_api_client = RebaseAPI(api_key = open("team_key.txt").read())
    current_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=7)
    WindGeneration=pd.DataFrame()
    for day in tqdm(pd.date_range("2024-03-15",current_day.strftime("%Y-%m-%d"))):
        day=day.strftime("%Y-%m-%d")
        wind_generation=rebase_api_client.get_variable(day=day,variable="wind_total_production")
        wind_generation["Wind_MWh_credit"]=0.5*wind_generation["generation_mw"]
        wind_generation["timestamp_utc"]=pd.to_datetime(wind_generation["timestamp_utc"],utc=True)
        WindGeneration=pd.concat([WindGeneration,wind_generation],axis=0)
    return WindGeneration

def getLatestPrice():
    rebase_api_client = RebaseAPI(api_key = open("team_key.txt").read())
    current_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=4)
    Prices=pd.DataFrame()
    for day in tqdm(pd.date_range("2024-03-15",current_day.strftime("%Y-%m-%d"))):
        day=day.strftime("%Y-%m-%d")
        DAP=rebase_api_client.get_variable(day=day,variable="day_ahead_price")
        SSP=rebase_api_client.get_variable(day=day,variable="imbalance_price")
        DAP["timestamp_utc"]=pd.to_datetime(DAP["timestamp_utc"],utc=True)
        DAP.rename(columns={"price":"dayahead_price"},inplace=True)
        SSP["timestamp_utc"]=pd.to_datetime(SSP["timestamp_utc"],utc=True)
        price=DAP.merge(SSP,how="inner",on="timestamp_utc")
        Prices=pd.concat([Prices,price],axis=0)

    return Prices

def getSolarDatasetfromNC(SolarGeneration,retain_time=False,is_train=True):

    current_day=pd.Timestamp.now(tz="UTC")-pd.Timedelta(days=1)
    modelling_table_solar=pd.DataFrame()
    for day in tqdm(pd.date_range("2024-03-15",current_day.strftime("%Y-%m-%d"))):
            
        dayad1=day+pd.Timedelta(days=1)
        day=day.strftime("%Y-%m-%d")
        dayad1=dayad1.strftime("%Y-%m-%d")
        try:
            latest_solar = xr.open_dataset(f"logs/weather/solar/ref0/{day}_latest_dwd_solar.nc")
        except:
            continue
    
        #Average solar radiation and cloud cover
        latest_solar_features=latest_solar[["SolarDownwardRadiation","CloudCover"]].mean(dim="point").to_dataframe().reset_index()
        
        #Maximum solar radiation and cloud cover
        latest_solar_features=latest_solar_features.merge(
            latest_solar[["SolarDownwardRadiation","CloudCover"]].max(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"maxSolarDownwardRadiation","CloudCover":"maxCloudCover","TotalPrecipitation":"maxTotalPrecipitation"}),
            how="left",on=["ref_datetime","valid_datetime"])
        
        #Minimum solar radiation and cloud cover
        latest_solar_features=latest_solar_features.merge(
            latest_solar[["SolarDownwardRadiation","CloudCover"]].min(dim="point").to_dataframe().reset_index().rename(columns={"SolarDownwardRadiation":"minSolarDownwardRadiation","CloudCover":"minCloudCover","TotalPrecipitation":"minTotalPrecipitation"}),
            how="left",on=["ref_datetime","valid_datetime"])
        
        #Convert to datetime
        latest_solar_features["ref_datetime"]=pd.to_datetime(latest_solar_features["ref_datetime"],utc=True)
        latest_solar_features["valid_datetime"]=pd.to_datetime(latest_solar_features["valid_datetime"],utc=True)
    
        #Filter out data points that are not within 47 hours
        latest_solar_features=latest_solar_features[(latest_solar_features["valid_datetime"]-latest_solar_features["ref_datetime"])<=pd.Timedelta(47,"h")]

        if is_train==False:
           latest_solar_features=latest_solar_features[(latest_solar_features["valid_datetime"]-latest_solar_features["ref_datetime"])>=pd.Timedelta(23,"h")] 
           
        #Interpolate 
        latest_solar_features=latest_solar_features.set_index("valid_datetime").resample("30T").interpolate("linear",limit=5).reset_index()
    
        #Extract features
        SolarFeatures=pd.DataFrame()
        for i in range(len(latest_solar_features)-2):
            feature={
            "rad_t-1_dwd":latest_solar_features.iloc[i]["SolarDownwardRadiation"],
            "rad_t-1_dwd_max":latest_solar_features.iloc[i]["maxSolarDownwardRadiation"],
            "rad_t-1_dwd_min":latest_solar_features.iloc[i]["minSolarDownwardRadiation"],
            
            "rad_t_dwd":latest_solar_features.iloc[i+1]["SolarDownwardRadiation"],
            "rad_t_dwd_max":latest_solar_features.iloc[i+1]["maxSolarDownwardRadiation"],
            "rad_t_dwd_min":latest_solar_features.iloc[i+1]["minSolarDownwardRadiation"],

            "rad_t+1_dwd":latest_solar_features.iloc[i+2]["SolarDownwardRadiation"],
            "rad_t+1_dwd_max":latest_solar_features.iloc[i+2]["maxSolarDownwardRadiation"],
            "rad_t+1_dwd_min":latest_solar_features.iloc[i+2]["minSolarDownwardRadiation"],
            
            "cloudcov_t-1_dwd":latest_solar_features.iloc[i]["CloudCover"],
            "cloudcov_t-1_dwd_max":latest_solar_features.iloc[i]["maxCloudCover"],
            "cloudcov_t-1_dwd_min":latest_solar_features.iloc[i]["minCloudCover"],
    
            "cloudcov_t_dwd":latest_solar_features.iloc[i+1]["CloudCover"],
            "cloudcov_t_dwd_max":latest_solar_features.iloc[i+1]["maxCloudCover"],
            "cloudcov_t_dwd_min":latest_solar_features.iloc[i+1]["minCloudCover"],
    
            "cloudcov_t+1_dwd":latest_solar_features.iloc[i+2]["CloudCover"],
            "cloudcov_t+1_dwd_max":latest_solar_features.iloc[i+2]["maxCloudCover"],
            "cloudcov_t+1_dwd_min":latest_solar_features.iloc[i+2]["minCloudCover"],
            
            "ref_datetime":latest_solar_features.iloc[i+1]["ref_datetime"],
            "valid_datetime":latest_solar_features.iloc[i+1]["valid_datetime"],
            }
            SolarFeatures=SolarFeatures._append(feature,ignore_index=True)
    
        modelling_table_solar=pd.concat([modelling_table_solar,SolarFeatures],axis=0)
    
    #Merge with solar generation data
    modelling_table_solar=modelling_table_solar.merge(SolarGeneration[["Solar_MWh_credit","timestamp_utc"]],how="inner",left_on="valid_datetime",right_on="timestamp_utc")

    modelling_table_solar["hours"]=pd.to_datetime(modelling_table_solar["valid_datetime"]).dt.hour
    columns_solar=pd.read_csv("data/dataset/full/dwd/SolarDataset.csv").columns.tolist()

    #Whether to retain time
    if retain_time==False:
        modelling_table_solar=modelling_table_solar[columns_solar]
    else:
        modelling_table_solar=modelling_table_solar[columns_solar+["timestamp_utc"]]

    #Delete missing values
    modelling_table_solar.dropna(inplace=True)
    
    return modelling_table_solar



# Convert nwp data frame to xarray
def weather_df_to_xr(weather_data):
  
  weather_data["ref_datetime"] = pd.to_datetime(weather_data["ref_datetime"],utc=True)
  weather_data["valid_datetime"] = pd.to_datetime(weather_data["valid_datetime"],utc=True)

  
  if 'point' in weather_data.columns:
    weather_data = weather_data.set_index(["ref_datetime",
                                          "valid_datetime",
                                          "point"])
  else:
      weather_data = pd.melt(weather_data,id_vars=["ref_datetime","valid_datetime"])
  
      weather_data = pd.concat([weather_data,
                            weather_data["variable"].str.split("_",expand=True)],
                            axis=1).drop(['variable',1,3], axis=1)
  
      weather_data.rename(columns={0:"variable",2:"latitude",4:"longitude"},inplace=True)
  
      weather_data = weather_data.set_index(["ref_datetime",
                                          "valid_datetime",
                                          "longitude",
                                          "latitude"])
      weather_data = weather_data.pivot(columns="variable",values="value")
  
  weather_data = weather_data.to_xarray()

  weather_data['ref_datetime'] = pd.DatetimeIndex(weather_data['ref_datetime'].values,tz="UTC")
  weather_data['valid_datetime'] = pd.DatetimeIndex(weather_data['valid_datetime'].values,tz="UTC")

  return weather_data

def day_ahead_market_times(today_date=pd.to_datetime('today')):

  tomorrow_date = today_date + pd.Timedelta(1,unit="day")
  DA_Market = [pd.Timestamp(datetime.datetime(today_date.year,today_date.month,today_date.day,23,0,0),
                          tz="Europe/London"),
              pd.Timestamp(datetime.datetime(tomorrow_date.year,tomorrow_date.month,tomorrow_date.day,22,30,0),
              tz="Europe/London")]

  DA_Market = pd.date_range(start=DA_Market[0], end=DA_Market[1],
                  freq=pd.Timedelta(30,unit="minute"))
  
  return DA_Market


def prep_submission_in_json_format(submission_data,market_day=pd.to_datetime('today') + pd.Timedelta(1,unit="day")):
  submission = []

  if any(submission_data["market_bid"]<0):
    submission_data.loc[submission_data["market_bid"]<0,"market_bid"] = 0
    warnings.warn("Warning...Some market bids were less than 0 and have been set to 0")

  if any(submission_data["market_bid"]>1800):
    submission_data.loc[submission_data["market_bid"]>1800,"market_bid"] = 1800
    warnings.warn("Warning...Some market bids were greater than 1800 and have been set to 1800")

  for i in range(len(submission_data.index)):
      submission.append({
          'timestamp': submission_data["datetime"][i].isoformat(),
          'market_bid': submission_data["market_bid"][i],
          'probabilistic_forecast': {
              10: submission_data["q10"][i],
              20: submission_data["q20"][i],
              30: submission_data["q30"][i],
              40: submission_data["q40"][i],
              50: submission_data["q50"][i],
              60: submission_data["q60"][i],
              70: submission_data["q70"][i],
              80: submission_data["q80"][i],
              90: submission_data["q90"][i],
          }
      })

  data = {
      'market_day': market_day.strftime("%Y-%m-%d"),
      'submission': submission
  }
  
  return data


def send_mes(recordings,resp,tomorrow):
    
  # Configuration
  mail_host = "xxx"
  mail_sender = "xxxxx"
  mail_license = "xxx"
  mail_receivers = ["xxxxx"]
  mm = MIMEMultipart('related')
  mm["From"] = "xxxxx"
  mm["To"] = "xxxxxx"
  
  subject_content = """HEFTcom24 Auto Submission"""
  mm["Subject"] = Header(subject_content,'utf-8')
  
  body_content = "response:"+str(resp.status_code)+"\n"+recordings
  message_text = MIMEText(body_content,"plain","utf-8")
  mm.attach(message_text)

  #Send images
  sendimagefile1 = open(f"logs/figs/{tomorrow}_forecast.png", 'rb').read()
  sendimagefile2 = open(f"logs/figs/{tomorrow}_benchmark_comp.png", 'rb').read()
  sendimagefile3 = open(f"logs/figs/{tomorrow}_bidding_comp.png", 'rb').read()
  image = MIMEImage(sendimagefile1)
  image.add_header('Content-ID', '<image1>')
  image["Content-Disposition"] = 'attachment; filename="forecast.png"'
  mm.attach(image)
  image = MIMEImage(sendimagefile2)
  image.add_header('Content-ID', '<image2>')
  image["Content-Disposition"] = 'attachment; filename="benchmark_comp.png"'
  mm.attach(image)
  image = MIMEImage(sendimagefile3)
  image.add_header('Content-ID', '<image3>')
  image["Content-Disposition"] = 'attachment; filename="bidding_comp.png"'
  mm.attach(image)

  #Send mail
  stp = smtplib.SMTP()
  stp.connect(mail_host, 25)  
  stp.set_debuglevel(1)
  stp.login(mail_sender,mail_license)
  stp.sendmail(mail_sender, mail_receivers, mm.as_string())
  print("Mail sent successfully")
  stp.quit()