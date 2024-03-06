from statsmodels.iolib import smpickle
import pickle
import numpy as np
#from _21_stacking import MetaMLP


# Define a function to adjust the forecast so that it is sorted from q10 to q90 for each time step
def adjust_forecast(forecast):
    quantiles = sorted(forecast.keys(), key=lambda x: int(x[1:]))
    forecast_array = np.array([forecast[q] for q in quantiles])
    forecast_array_sorted = np.sort(forecast_array, axis=0)
    for i, q in enumerate(quantiles):
        forecast[q] = forecast_array_sorted[i, :]
    
    return forecast

def forecast(wind_features,solar_features,Dataset_stats,hours,full,model_name,WLimit,maxPower=415):

    #--------------------------------加载模型---------------------------
    Models_wind={}
    Models_solar={}
    for quantile in range(10,100,10):
        if full:
            with open(f"models/{model_name}/full/wind_q{quantile}.pkl","rb") as f:
                Models_wind[f"q{quantile}"]=pickle.load(f)
            with open(f"models/{model_name}/full/solar_q{quantile}.pkl","rb") as f:
                Models_solar[f"q{quantile}"]=pickle.load(f)
        else:
            with open(f"models/{model_name}/partial/wind_q{quantile}.pkl","rb") as f:
                Models_wind[f"q{quantile}"]=pickle.load(f)
            with open(f"models/{model_name}/partial/solar_q{quantile}.pkl","rb") as f:
                Models_solar[f"q{quantile}"]=pickle.load(f)
    
    #加载限电预测模型
    Models_wind_remit={}
    for quantile in range(10,100,10):
        with open(f"models/LGBM/full/wind_remit_q{quantile}.pkl","rb") as f:
            Models_wind_remit[f"q{quantile}"]=pickle.load(f)
    
    #--------------------------------预测---------------------------------
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    Total_generation_forecast={}
    for idx,quantile in enumerate(range(10,100,10)):
        
        #前向
        output_wind= Models_wind[f"q{quantile}"].predict(wind_features)
        output_solar= Models_solar[f"q{quantile}"].predict(solar_features)

        #逆归一化
        output_wind=output_wind*Dataset_stats["Std"]["labels"]["wind"]+Dataset_stats["Mean"]["labels"]["wind"]
        output_solar=output_solar*Dataset_stats["Std"]["labels"]["solar"]+Dataset_stats["Mean"]["labels"]["solar"]

        #负值清0
        output_wind[output_wind<0]=0
        output_solar[output_solar<1e-2]=0
        
        #夜晚光伏全补0
        output_solar[hours<=2]=0
        output_solar[hours>=21]=0


        Wind_Generation_Forecast[f"q{quantile}"]=output_wind
        Solar_Generation_Forecast[f"q{quantile}"]=output_solar
        
    #风电限电
    if WLimit:
        
        '''
        original_forecast=Wind_Generation_Forecast[f"q{40}"].copy()
        maxpowers=[360,370,380,390,400,405,410,412.5,415]
        for quantile in range(10,100,10):
            #获取限电点
            idxs_limit=original_forecast>maxpowers[quantile//10-1]
            if np.any(idxs_limit):
                Wind_Generation_Forecast[f"q{quantile}"][idxs_limit]=maxpowers[quantile//10-1]
            
            #限制最高功率
            Wind_Generation_Forecast[f"q{quantile}"][Wind_Generation_Forecast[f"q{quantile}"]>maxpowers[quantile//10-1]]=maxpowers[quantile//10-1]
        '''
    
        idxs_limit=Wind_Generation_Forecast["q50"]>maxPower #获取可能限电的点

        #如果idxs_limit包含true
        if np.any(idxs_limit):
            for quantile in range(10,100,10):
                #调用限电模型预测
                quantiles = sorted(Wind_Generation_Forecast.keys(), key=lambda x: int(x[1:]))
                forecast_array = np.array([Wind_Generation_Forecast[q] for q in quantiles]).T
                Wind_Generation_Forecast[f"q{quantile}"][idxs_limit]=Models_wind_remit[f"q{quantile}"].predict(forecast_array[idxs_limit])
                #整体限电功率上限
                Wind_Generation_Forecast[f"q{quantile}"][Wind_Generation_Forecast[f"q{quantile}"]>maxPower]=maxPower
  
        
    #汇总发电数据
    for quantile in range(10,100,10):
        Total_generation_forecast[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]+Solar_Generation_Forecast[f"q{quantile}"]

    #分位数重新排序，确保大的分位数结果更大
    Total_generation_forecast=adjust_forecast(Total_generation_forecast)
    
    
    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast




def forecastByBenchmark(wind_forecast_table,solar_forecat_table):

    #--------------------------------加载模型---------------------------
    Models_benchmark_wind={}
    Models_benchmark_solar={}
    for quantile in range(10,100,10):
        with open(f"models/benchmark/wind_q{quantile}.pickle", 'rb') as f:
            mod_wind = smpickle.load_pickle(f)
        with open(f"models/benchmark/solar_q{quantile}.pickle", 'rb') as f:
            mod_solar = smpickle.load_pickle(f)
        Models_benchmark_wind[f"q{quantile}"]=mod_wind
        Models_benchmark_solar[f"q{quantile}"]=mod_solar

    #--------------------------------预测---------------------------------
    Total_generation_forecast={}
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    for quantile in range(10,100,10):
        
        #前向
        output_wind=Models_benchmark_wind[f"q{quantile}"].predict(wind_forecast_table)
        output_solar=Models_benchmark_solar[f"q{quantile}"].predict(solar_forecat_table)
        
        #手动修正
        output_wind[output_wind<0]=0
        output_solar[output_solar<1e-2]=0
        

        #汇总结果
        total_generation_forecast=output_wind+output_solar
        #total_generation_forecast=[elem.item() for elem in total_generation_forecast]

        Total_generation_forecast[f"q{quantile}"]=total_generation_forecast
        Wind_Generation_Forecast[f"q{quantile}"]=output_wind
        Solar_Generation_Forecast[f"q{quantile}"]=output_solar

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast


def forecastByStacking(wind_features,solar_features,Dataset_stats,hours,full):
    
    #--------------------------------加载模型---------------------------
    Regressor_names=["LGBM","CatBoost","MLP","ExtraTrees"]
    Models_wind=[]
    Models_solar=[]
    for regressor_name in Regressor_names:
        with open(f"models/stacking/{regressor_name}/wind.pkl","rb") as f:
            model=pickle.load(f)
        Models_wind.append(model)
        with open(f"models/stacking/{regressor_name}/solar.pkl","rb") as f:
            model=pickle.load(f)
        Models_solar.append(model)

    #--------------------------------加载元学习器模型---------------------------
    Models_wind_meta={}
    Models_solar_meta={}
    for quantile in range(10,100,10):
        with open(f"models/stacking/meta/wind_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        Models_wind_meta[f"q{quantile}"]=model
        with open(f"models/stacking/meta/solar_q{quantile}.pkl","rb") as f:
            model=pickle.load(f)
        Models_solar_meta[f"q{quantile}"]=model


    #--------------------------------预测---------------------------------
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    Total_generation_forecast={}
    for quantile in range(10,100,10):

        
        output_wind=[]
        output_solar=[]
        
        #依次预测
        for model in Models_wind:
            output_wind.append(model.predict(wind_features))
            
        for model in Models_solar:
            output_solar.append(model.predict(solar_features))

        '''
        #加载Benchmark模型的预测结果
        with open("models/benchmark/wind_q50.pickle", 'rb') as f:
            mod_benchmark_wind= smpickle.load_pickle(f)

        with open("models/benchmark/solar_q50.pickle", 'rb') as f:
            mod_benchmark_solar= smpickle.load_pickle(f)

        output_wind.append(mod_benchmark_wind.predict(wind_forecast_table))
        output_solar.append(mod_benchmark_solar.predict(solar_forecat_table))
        '''
        
        output_wind=np.stack(output_wind,axis=-1)
        output_solar=np.stack(output_solar,axis=-1)
        

        #逆归一化
        output_wind=output_wind*Dataset_stats["Std"]["labels"]["wind"]+Dataset_stats["Mean"]["labels"]["wind"]
        output_solar=output_solar*Dataset_stats["Std"]["labels"]["solar"]+Dataset_stats["Mean"]["labels"]["solar"]

        #元学习器预测
        output_wind=Models_wind_meta[f"q{quantile}"].predict(output_wind)
        output_solar=Models_solar_meta[f"q{quantile}"].predict(output_solar)

        #手动修正
        output_wind[output_wind<0]=0
        output_solar[output_solar<1e-2]=0
        
        #夜晚光伏全补0
        output_solar[hours<=2]=0
        output_solar[hours>=21]=0


        #汇总结果
        total_generation_forecast=output_wind+output_solar
        total_generation_forecast=[elem.item() for elem in total_generation_forecast]

        Total_generation_forecast[f"q{quantile}"]=total_generation_forecast
        Wind_Generation_Forecast[f"q{quantile}"]=output_wind
        Solar_Generation_Forecast[f"q{quantile}"]=output_solar

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast







