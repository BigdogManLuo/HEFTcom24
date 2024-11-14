import numpy as np
import matplotlib.pyplot as plt
import pickle
import comp_utils
from comp_utils import pinball
import matplotlib.cm as cm


rebase_api_client = comp_utils.RebaseAPI(api_key = open("team_key.txt").read())
day="2024-05-03"

#======================Get History Submission Data=============================
submissions=rebase_api_client.get_submissions(market_day=day)
solution=submissions["items"][-1]["solution"]
print(f"market day: {solution['market_day']}")
submission=solution['submission']

Total_Generation_Forecast={f"q{quantile}":[] for quantile in range(10,100,10)}

for quantile in range(10,100,10):
    for i in range(48):
        Total_Generation_Forecast[f"q{quantile}"].append(submission[i]["probabilistic_forecast"][f'{quantile}'])
    Total_Generation_Forecast[f"q{quantile}"]=np.array(Total_Generation_Forecast[f"q{quantile}"])

with open(f"logs/forecast/{day}_Wind_Generation_Forecast.pkl","rb") as f:
    Wind_Generation_Forecast=pickle.load(f)

with open(f"logs/forecast/{day}_Solar_Generation_Forecast.pkl","rb") as f:
    Solar_Generation_Forecast=pickle.load(f)

#=======================Get Actual Generation===========================================

solar_generation=rebase_api_client.get_variable(day=day,variable="solar_total_production")
solar_generation["Solar_MWh_Credit"]=0.5*solar_generation["generation_mw"]
solar_generation=solar_generation[0:48]

wind_generation=rebase_api_client.get_variable(day=day,variable="wind_total_production")
wind_generation["Wind_MWh_Credit"]=0.5*wind_generation["generation_mw"]-wind_generation["boa"]

Total_Generation_True=wind_generation["Wind_MWh_Credit"]+solar_generation["Solar_MWh_Credit"]
Total_Generation_True=Total_Generation_True[0:48]

#Calculate pinball loss
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=Total_Generation_True,y_hat=Total_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"pinball_score: {pinball_score}")

#Calculate Wind pinball loss
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=wind_generation["Wind_MWh_Credit"],y_hat=Wind_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"Wind pinball_score: {pinball_score}")

#Calculate Solar pinball loss
pinball_score=0
for quantile in range(10,100,10):
    pinball_score+=pinball(y=solar_generation["Solar_MWh_Credit"],y_hat=Solar_Generation_Forecast[f"q{quantile}"],alpha=quantile/100).mean()
pinball_score=pinball_score/9
print(f"Solar pinball_score: {pinball_score}")


#Get Price Data
DAP=rebase_api_client.get_variable(day=day,variable="day_ahead_price")
SSP=rebase_api_client.get_variable(day=day,variable="imbalance_price")

#Calculate Revenue
Revenue_sub=np.zeros(48)
for i in range(48):
    bid=int(submission[i]['market_bid'])
    Revenue_sub[i]=DAP['price'].values[i]*bid+(Total_Generation_True[i]-bid)*SSP['imbalance_price'].values[i]-0.07*(Total_Generation_True[i]-bid)**2
print(f"revenue_{day}: {Revenue_sub.sum()}")

#If the forecast data with quantile=50 is used as the bid amount
revenue=0
Revenue_q50=np.zeros(48)
for i in range(48):
    bid=submission[i]["probabilistic_forecast"]["50"]
    Revenue_q50[i]=DAP['price'].values[i]*bid+(Total_Generation_True[i]-bid)*SSP['imbalance_price'].values[i]-0.07*(Total_Generation_True[i]-bid)**2   
print(f"revenue_{day}_q50: {Revenue_q50.sum()}")

plt.plot(Revenue_sub,label="submission")
plt.plot(Revenue_q50,label="q50")
plt.legend()


#======================Compare Forecast and True Generation=============================
cmap = cm.get_cmap('gray_r')
colors = cmap(np.abs(np.linspace(-1, 1, 9)**2))
plt.figure(figsize=(12,6))
for quantile in range(10,100,10):
    plt.plot(Total_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}",color=colors[quantile//10-1])
    
plt.plot(Total_Generation_True,label="True",marker=".")
plt.legend()
plt.title(f"{solution['market_day']}")
plt.grid()
plt.show()

plt.figure(figsize=(12,6))
for quantile in range(10,100,10):
    plt.plot(Wind_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}",color=colors[quantile//10-1])
plt.plot(wind_generation["Wind_MWh_Credit"],label="True",marker=".")
plt.legend()
plt.title("Wind")
plt.grid()
plt.show()

cmap = cm.get_cmap('gray_r')
colors = cmap(np.abs(np.linspace(-1, 1, 9)**2))
plt.figure(figsize=(12,6))
for quantile in range(10,100,10):
    plt.plot(Solar_Generation_Forecast[f"q{quantile}"],label=f"q{quantile}",color=colors[quantile//10-1])
plt.plot(solar_generation["Solar_MWh_Credit"],label="True",marker=".")
plt.legend()
plt.title("Solar")
plt.grid()
plt.show()


