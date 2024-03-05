from requests import Session
import requests
import pandas as pd
import datetime
import warnings
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from email.mime.image import MIMEImage


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

    variables = 'SolarDownwardRadiation, CloudCover, Temperature'
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
    
    #记录每次提交的代码改动记录
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
  


def pinball(y,y_hat,alpha):
    '''
    分位数预测损失函数
    y:真实值 (batch_size,1) tensor
    y_hat:预测值 (batch_size,1) tensor
    alpha:分位数 0.1,0.2,...,0.9
    '''
    return ((y-y_hat)*alpha*(y>=y_hat) + (y_hat-y)*(1-alpha)*(y<y_hat)).mean()



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
    
  #配置基础信息
  mail_host = "smtp.qq.com"
  mail_sender = "852046386@qq.com"
  mail_license = "alfketoxyyjhbcdf"
  mail_receivers = ["sashabanks@126.com"]
  mm = MIMEMultipart('related')
  mm["From"] = "sender_name<852046386@qq.com>"
  mm["To"] = "receiver_1_name<sashabanks@126.com>"
  
  # 邮件主题
  subject_content = """HEFTcom24 自动提交"""
  mm["Subject"] = Header(subject_content,'utf-8')
  
  # 邮件正文
  body_content = "response:"+str(resp.status_code)+"\n"+recordings
  message_text = MIMEText(body_content,"plain","utf-8")
  mm.attach(message_text)
  
  #读取并发送图片
  sendimagefile1 = open(f"logs/figs/{tomorrow}_forecast.png", 'rb').read()
  sendimagefile2 = open(f"logs/figs/{tomorrow}_benchmark_comp.png", 'rb').read()
  image = MIMEImage(sendimagefile1)
  image.add_header('Content-ID', '<image1>')
  image["Content-Disposition"] = 'attachment; filename="forecast.png"'
  mm.attach(image)
  image = MIMEImage(sendimagefile2)
  image.add_header('Content-ID', '<image2>')
  image["Content-Disposition"] = 'attachment; filename="benchmark_comp.png"'
  mm.attach(image)
  


  #发送邮件
  stp = smtplib.SMTP()
  stp.connect(mail_host, 25)  
  stp.set_debuglevel(1)
  stp.login(mail_sender,mail_license)
  stp.sendmail(mail_sender, mail_receivers, mm.as_string())
  print("邮件发送成功")
  stp.quit()
