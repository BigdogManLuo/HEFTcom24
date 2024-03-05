import torch
from _13_train_NN import WindMLP,SolarMLP
import statsmodels.formula.api as smf
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

def forecast(wind_features,solar_features,Dataset_stats,hours,full,model_name,WLimit,maxPower=215):

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
    
    #加载meta模型
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
        
        original_forecast=Wind_Generation_Forecast[f"q{30}"].copy()
        maxpowers=[360,370,380,390,400,405,410,412.5,415]
        for quantile in range(10,100,10):
            #获取限电点
            idxs_limit=original_forecast>maxpowers[quantile//10-1]
            if np.any(idxs_limit):
                Wind_Generation_Forecast[f"q{quantile}"][idxs_limit]=maxpowers[quantile//10-1]
            
            #限制最高功率
            Wind_Generation_Forecast[f"q{quantile}"][Wind_Generation_Forecast[f"q{quantile}"]>maxpowers[quantile//10-1]]=maxpowers[quantile//10-1]
        
    
        '''
        #旧策略失效
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
                #限电功率下限
                #elements = Wind_Generation_Forecast[f"q{quantile}"][idxs_limit]  
                #elements[np.where(elements < 190)] = 190
                #Wind_Generation_Forecast[f"q{quantile}"][idxs_limit] = elements
        '''
        
    #汇总发电数据
    for quantile in range(10,100,10):
        Total_generation_forecast[f"q{quantile}"]=Wind_Generation_Forecast[f"q{quantile}"]+Solar_Generation_Forecast[f"q{quantile}"]

    #分位数重新排序，确保大的分位数结果更大
    Total_generation_forecast=adjust_forecast(Total_generation_forecast)
    
    
    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast



def forecastByNN(wind_features,solar_features,Dataset_stats,full):
    """
    调用神经网络模型产生预测结果
    """

    #--------------------------------加载模型---------------------------
    Models_wind={}
    Models_solar={}

    #超参数
    hidden_size_wind=32
    hidden_size_solar=16

    #GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    for quantile in range(10,100,10):

        #初始化模型
        model_wind=WindMLP(input_size=wind_features.shape[1],
                    hidden_size=hidden_size_wind,
                    output_size=1).to(device=device)

        model_solar=SolarMLP(input_size=solar_features.shape[1],
                    hidden_size=hidden_size_solar,
                    output_size=1).to(device=device)

        if full:
            #加载权重
            model_wind.load_state_dict(torch.load(f"models/NN/full/wind_q{quantile}.pt"))
            model_solar.load_state_dict(torch.load(f"models/NN/full/solar_q{quantile}.pt"))
        else:
            #加载权重
            model_wind.load_state_dict(torch.load(f"models/NN/partial/wind_q{quantile}.pt"))
            model_solar.load_state_dict(torch.load(f"models/NN/partial/solar_q{quantile}.pt"))
    
        Models_wind[f"q{quantile}"]=model_wind
        Models_solar[f"q{quantile}"]=model_solar


    #--------------------------------预测---------------------------------
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    Total_generation_forecast={}
    with torch.no_grad():
        for quantile in range(10,100,10):
            
            #前向传播
            pred_wind=Models_wind[f"q{quantile}"](torch.tensor(wind_features,dtype=torch.float32,device=device))
            pred_solar=Models_solar[f"q{quantile}"](torch.tensor(solar_features,dtype=torch.float32,device=device))

            #反标准化
            Std_labels_wind=torch.tensor(Dataset_stats["Std"]["labels"]["wind"],device=device)
            Mean_labels_wind=torch.tensor(Dataset_stats["Mean"]["labels"]["wind"],device=device)
            Std_labels_solar=torch.tensor(Dataset_stats["Std"]["labels"]["solar"],device=device)
            Mean_labels_solar=torch.tensor(Dataset_stats["Mean"]["labels"]["solar"],device=device)
            
            pred_wind=pred_wind*Std_labels_wind+Mean_labels_wind
            pred_solar=pred_solar*Std_labels_solar+Mean_labels_solar
            
            #手动补偿
            pred_solar[pred_solar<1e-2]=0
            pred_wind[pred_wind<0]=0

            #汇总结果
            total_generation_forecast=pred_wind.cpu().numpy()+pred_solar.cpu().numpy()
            total_generation_forecast=[elem.item() for elem in total_generation_forecast]

            Total_generation_forecast[f"q{quantile}"]=total_generation_forecast
            Wind_Generation_Forecast[f"q{quantile}"]=pred_wind.cpu().numpy()
            Solar_Generation_Forecast[f"q{quantile}"]=pred_solar.cpu().numpy()
    
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

'''
def forecastByStacking(wind_features,solar_features,Dataset_stats,hours,full):
    
    #--------------------------------加载模型---------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Models_wind={}
    Models_solar={}
    Models_wind_meta={}
    Models_solar_meta={}
    for quantile in range(10,100,10):

        #加载LGMB模型
        if full:
            bst_wind = lgb.Booster(model_file=f"models/LGBM/full/wind_q{quantile}.txt")
            bst_solar = lgb.Booster(model_file=f"models/LGBM/full/solar_q{quantile}.txt")
        else:
            bst_wind = lgb.Booster(model_file=f"models/LGBM/partial/wind_q{quantile}.txt")
            bst_solar = lgb.Booster(model_file=f"models/LGBM/partial/solar_q{quantile}.txt")

        #加载CatBoost模型
        if full:
            with open(f"models/CatBoost/full/wind_q{quantile}.pkl","rb") as f:
                cb_wind=pickle.load(f)
            with open(f"models/CatBoost/full/solar_q{quantile}.pkl","rb") as f:
                cb_solar=pickle.load(f)
        else:
            with open(f"models/CatBoost/partial/wind_q{quantile}.pkl","rb") as f:
                cb_wind=pickle.load(f)
            with open(f"models/CatBoost/partial/solar_q{quantile}.pkl","rb") as f:
                cb_solar=pickle.load(f)

        Models_wind[f"q{quantile}"]= [bst_wind,cb_wind]
        Models_solar[f"q{quantile}"]= [bst_solar,cb_solar]

        #加载元学习器
        model_wind_meta=MetaMLP(input_size=2)
        model_solar_meta=MetaMLP(input_size=2)
        model_wind_meta.load_state_dict(torch.load(f"models/stacking/{'wind'}_q{quantile}.pt"))
        model_solar_meta.load_state_dict(torch.load(f"models/stacking/{'solar'}_q{quantile}.pt"))
        
        Models_wind_meta[f"q{quantile}"]=model_wind_meta
        Models_solar_meta[f"q{quantile}"]=model_solar_meta


    #--------------------------------预测---------------------------------
    Wind_Generation_Forecast={}
    Solar_Generation_Forecast={}
    Total_generation_forecast={}
    for quantile in range(10,100,10):

        #依次预测
        output_wind=[]
        output_solar=[]
        for model in Models_wind[f"q{quantile}"]:
            output_wind.append(model.predict(wind_features))
        for model in Models_solar[f"q{quantile}"]:
            output_solar.append(model.predict(solar_features))

        output_wind=np.stack(output_wind,axis=-1)
        output_solar=np.stack(output_solar,axis=-1)
        
        #逆归一化
        output_wind=output_wind*Dataset_stats["Std"]["labels"]["wind"]+Dataset_stats["Mean"]["labels"]["wind"]
        output_solar=output_solar*Dataset_stats["Std"]["labels"]["solar"]+Dataset_stats["Mean"]["labels"]["solar"]

        #output_wind=np.column_stack((hours,output_wind))
        #output_solar=np.column_stack((hours,output_solar))

        #元学习器预测
        output_wind=Models_wind_meta[f"q{quantile}"](torch.tensor(output_wind,dtype=torch.float32,device=device))
        output_solar=Models_solar_meta[f"q{quantile}"](torch.tensor(output_solar,dtype=torch.float32,device=device))

        #手动修正
        output_wind[output_wind<0]=0
        output_solar[output_solar<1e-2]=0
        
        #转换为numpy
        output_wind=output_wind.detach().cpu().numpy()
        output_solar=output_solar.detach().cpu().numpy()

        #汇总结果
        total_generation_forecast=output_wind+output_solar
        total_generation_forecast=[elem.item() for elem in total_generation_forecast]

        Total_generation_forecast[f"q{quantile}"]=total_generation_forecast
        Wind_Generation_Forecast[f"q{quantile}"]=output_wind
        Solar_Generation_Forecast[f"q{quantile}"]=output_solar

    return Total_generation_forecast,Wind_Generation_Forecast,Solar_Generation_Forecast
'''







