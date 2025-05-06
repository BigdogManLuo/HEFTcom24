import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import utils
import utils_forecasting
from tqdm import tqdm
import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from matplotlib import pyplot as plt
import scienceplots
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from matplotlib.ticker import MaxNLocator
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX


# 自动选择最优阶数 (BIC准则倾向简化模型)
def  autoParams(ts):
    arma_model = auto_arima(
        ts,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        seasonal=False,
        information_criterion='bic',  # 比AIC更严格
        stepwise=True,
        suppress_warnings=True
    )
    optimal_order = arma_model.order
    
    return optimal_order


def check_time_regularity(df, time_col='dtm'):

    time_series = df[time_col].sort_values().reset_index(drop=True)
    
    deltas = time_series.diff().dropna()
    
    unique_deltas = deltas.unique()
    
    if len(unique_deltas) == 1:
        return True, unique_deltas[0]
    else:
        return False, None

def testStationary(ts):
    
    stl = STL(ts, period=48).fit()
    stl.plot()

    #ADF Stationarity Test
    adf_result = adfuller(ts)
    print(f"ADF Statistic: {adf_result[0]}")
    
    plt.figure(figsize=(12,6))
    plt.plot(ts)
    plt.axhline(ts.mean(), color='r', linestyle='--', label='Mean')
    plt.title('Stationary Series Verification')
    plt.show()
    
    kpss_stat, kpss_p, _, _ = kpss(ts)
    print(f"KPSS p-value: {kpss_p}")  # >0.05则确认平稳


def plot_academic_decomposition(df, figsize=(12, 9), 
                                save_path=None, dpi=660):
    
    df=df[df["Price_diff"]>-400]

    # 设置学术图表样式
    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 25,  # 调大字体大小
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 2.5,  # 加粗坐标轴
        'xtick.major.width': 2.5,  # 加粗x轴刻度
        'ytick.major.width': 2.5,  # 加粗y轴刻度
        'savefig.dpi': dpi,
        'figure.autolayout': True
    })

    # 创建画布和子图
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    try:
        # STL分解
        stl = STL(df.set_index('dtm')['Price_diff'], period=48, robust=True).fit()
    except Exception as e:
        raise ValueError(f"STL分解失败: {str(e)}") from e

    # 原始序列
    axes[0].plot(df['dtm'], df['Price_diff'], 
                color='black',alpha=0.8, linewidth=1.5, label='Observed')
    axes[0].set_title('(a) Original Time Series', y=0.85, x=0.02, ha='left', fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].set_ylabel('Value', labelpad=10)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1)


    # 趋势项
    axes[1].plot(df['dtm'], stl.trend, 
                color='black',alpha=0.8, linewidth=2, label='Trend')
    axes[1].set_title('(b) Trend Component', y=0.85, x=0.02, ha='left', fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].set_ylabel('Trend', labelpad=10)

    # 季节项
    axes[2].plot(df['dtm'], stl.seasonal, 
                color='black',alpha=0.8, linewidth=1, label='Seasonal')
    axes[2].set_title('(c) Seasonal Component', y=0.85, x=0.02, ha='left', fontweight='bold')
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].set_ylabel('Seasonality', labelpad=10)
    axes[2].set_xlabel('Date', labelpad=10)

    # 日期格式优化
    date_fmt = mdates.DateFormatter('%Y-%m')
    axes[2].xaxis.set_major_formatter(date_fmt)
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # y轴刻度优化
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(prune='both', nbins=6))

    # 保存图表
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    
    plt.show()


def testTrading(pathtype):

    alpha=0.1
    # Load Data
    features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype=pathtype,source="dwd")
    features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype=pathtype,source="gfs")
    IntegratedDataset=pd.read_csv(f"../data/dataset/{pathtype}/IntegratedDataset.csv")

    #Energy Data
    if pathtype=="test":
        energy_data=pd.read_csv("../data/raw/Energy_Data_20200920_20240118.csv")
    
    elif pathtype=="latest":
        energy_data1=pd.read_csv("../data/raw/Energy_Data_20200920_20240118.csv")
        energy_data2=pd.read_csv("../data/raw/Energy_Data_20240119_20240519.csv")
        energy_data=pd.concat([energy_data1,energy_data2],axis=0)

    energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
    energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
    energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]
    energy_data["hours"]=energy_data["dtm"].dt.hour
    energy_data["Price_diff"]=energy_data["DA_Price"]-energy_data["SS_Price"]
    energy_data["Total_MWh"]=energy_data["Wind_MWh_credit"]+energy_data["Solar_MWh_credit"]

    #Modelling table
    modelling_table=energy_data[["dtm","hours","Total_MWh","SS_Price","DA_Price","Price_diff"]].reset_index(drop=True)
    modelling_table=modelling_table[modelling_table["dtm"].isin(IntegratedDataset["valid_datetime"])]

    
    # Stationary Test 
    testStationary(modelling_table["Price_diff"])
    
    #Visualization
    plot_academic_decomposition(modelling_table,save_path=f"../figs/price_spread_original_{pathtype}")

    if pathtype=="test":

        params_q50={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "hours":features_solar_dwd[:,-1],
            "full":False,
            "WLimit":False,
            "SolarRevise":False,
            "aggregation":True
        }
    
        params_MSE={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "full":False,
            "hours":modelling_table["hours"].values,
            "WLimit":False,
            "SolarRevise":False
        }

    elif pathtype=="latest":

        params_q50={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "hours":features_solar_dwd[:,-1],
            "full":True,
            "WLimit":True,
            "SolarRevise":True,
            "rolling_test":True,
            "availableCapacities":IntegratedDataset["availableCapacity"].values,
            "aggregation":True
        }

        params_MSE={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "full":True,
            "hours":modelling_table["hours"].values,
            "WLimit":True,
            "availableCapacity":IntegratedDataset["availableCapacity"].values,
            "SolarRevise":True,
            "rolling_test":True,
        }
    
    Total_generation_forecast_q50,_,_=utils_forecasting.forecast_total(**params_q50)
    modelling_table["q50"]=Total_generation_forecast_q50["q50"]
    modelling_table["mse"]=utils_forecasting.forecast_bidding(**params_MSE).values

    # Benchmarks
    modelling_table["pd_pred"]=np.array([0]*len(modelling_table))
    modelling_table["pd_pred_persistance"]=np.array([0]*len(modelling_table))  # Persistance model
    modelling_table["pd_pred_persistance_seasonal"]=np.array([0]*len(modelling_table))  # Seasonal Persistance model
    modelling_table["pd_pred_persistance_average"]=np.array([0]*len(modelling_table))  # Average Persistance model
    modelling_table["pd_pred_ETS"]=np.array([0]*len(modelling_table))  # ETS model
    modelling_table["pd_pred_arma"]=np.array([0]*len(modelling_table)) # ARIMA model
    modelling_table["pd_pred_ar"]=np.array([0]*len(modelling_table))  # AR model
    modelling_table["pd_pred_sarimax"]=np.array([0]*len(modelling_table))  # SARIMAX model
 
    #=========================================Test Trading======================================================
    for day in tqdm(pd.date_range(modelling_table["dtm"].min()-pd.Timedelta(days=64),modelling_table["dtm"].max()-pd.Timedelta(days=64))):

        start_day=day
        end_day=start_day+pd.Timedelta(days=60)
        target_day=end_day+pd.Timedelta(days=4)
        
        past_df=energy_data[(energy_data["dtm"]>=start_day) & (energy_data["dtm"]<=end_day)]
        fureture_time_range=(modelling_table["dtm"] >= target_day) & (modelling_table["dtm"] <= target_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=30))
        
        # ETSModel
        model = ExponentialSmoothing(
            past_df["Price_diff"].values[:-1],
            initialization_method="heuristic",
            trend=None,  
            seasonal='add',
            seasonal_periods=48,
            damped_trend=False,
            use_boxcox=False
        )
        model_fit = model.fit(smoothing_level=0.4,smoothing_seasonal=0.3,optimized=False)
        forecast = model_fit.forecast(steps=len(fureture_time_range[fureture_time_range==True]))
        modelling_table.loc[fureture_time_range, "pd_pred_ETS"] = forecast*alpha
        
        
        # ARMA Model
        model = ARIMA(
            past_df["Price_diff"].values[:-1], 
            order=(0, 0, 3),
            enforce_stationarity=True,
            enforce_invertibility=True,
            trend='c')
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(fureture_time_range[fureture_time_range==True]))
        modelling_table.loc[fureture_time_range, "pd_pred_arma"] = forecast*alpha


        # AR Model
        model = ARIMA(past_df["Price_diff"].values[:-1], 
                      order=(1, 0, 0),
                      enforce_stationarity=True,
                      enforce_invertibility=True,
                      trend='c')
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(fureture_time_range[fureture_time_range==True]))
        modelling_table.loc[fureture_time_range, "pd_pred_ar"] = forecast*alpha

        # Persistance Model 
        modelling_table.loc[fureture_time_range, "pd_pred_persistance"] = past_df["Price_diff"].iloc[-1]
        
        # Average Persistance Model
        modelling_table.loc[fureture_time_range, "pd_pred_persistance_average"] = past_df["Price_diff"].mean()

        # Seasonal Persistance Model
        modelling_table.loc[fureture_time_range, "pd_pred_persistance_seasonal"] = past_df["Price_diff"][-len(fureture_time_range[fureture_time_range==True]):].values*alpha
        
        # SARIMAX Model
        model = SARIMAX(
            endog=past_df['Price_diff'].values[:-1],
            exog=past_df['hours'].values[:-1],
            order=(1,0,1),  # ARIMA(p,d,q)
        )
        model_fit = model.fit(disp=False)
        forecast = model_fit.get_forecast(
            steps=len(fureture_time_range[fureture_time_range==True]),
            exog=modelling_table.loc[fureture_time_range, "hours"].values
        )
        modelling_table.loc[fureture_time_range, "pd_pred_sarimax"] = forecast.predicted_mean*alpha
        
        # Stochastic Trading
        pd_in_hours=past_df["Price_diff"].groupby(past_df["hours"]).mean()
        target_df=modelling_table.loc[(modelling_table["dtm"]>=target_day) & (modelling_table["dtm"]<=target_day+pd.Timedelta(days=1)-pd.Timedelta(minutes=30))]
        modelling_table.loc[fureture_time_range, "pd_pred"] = pd_in_hours[target_df["hours"].values].values
        
    
    return modelling_table

def calculateBidding(modelling_table,methods_list):

    modelling_table["bidding_q50pd"]=modelling_table["q50"]+7*modelling_table["pd_pred"]
    modelling_table["bidding_msepd"]=modelling_table["mse"]+7*modelling_table["pd_pred"]
    modelling_table.loc[modelling_table["bidding_q50pd"]<0,"bidding_q50pd"]=0
    modelling_table.loc[modelling_table["bidding_q50pd"]>1800,"bidding_q50pd"]=1800
    modelling_table.loc[modelling_table["bidding_msepd"]<0,"bidding_msepd"]=0
    modelling_table.loc[modelling_table["bidding_msepd"]>1800,"bidding_msepd"]=1800


    for method in methods_list:
        modelling_table["bidding_"+method]=modelling_table["mse"]+7*modelling_table["pd_pred_"+method]
        modelling_table.loc[modelling_table[f"bidding_{method}"]<0,f"bidding_{method}"]=0
        modelling_table.loc[modelling_table[f"bidding_{method}"]>1800,f"bidding_{method}"]=1800

    return modelling_table

def showResults(modelling_table,methods_list,pathtype):

    #1：q50 
    R1=utils.getRevenue(modelling_table["q50"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])

    #2：mse
    R2=utils.getRevenue(modelling_table["mse"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])

    #3：ST(q50)
    R3=utils.getRevenue(modelling_table["bidding_q50pd"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])

    #4：ST(mse)
    R4=utils.getRevenue(modelling_table["bidding_msepd"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])

    Revenues=dict()
    Revenues["q50"]=R1
    Revenues["mse"]=R2
    Revenues["ST(q50)"]=R3
    Revenues["ST(mse)"]=R4

    for method in methods_list:
        Revenues[method]=utils.getRevenue(modelling_table[f"bidding_{method}"],modelling_table["Total_MWh"],modelling_table["DA_Price"],modelling_table["SS_Price"])

    for method in Revenues.keys():
        print(f"Revenue {method}: ", Revenues[method].sum())

    # Save Revenue
    if pathtype=="test":
        caseNum="case1"
    elif pathtype=="latest":
        caseNum="case2"
        
    if not os.path.exists(f"../data/revenues/{caseNum}"):
        os.makedirs(f"../data/revenues/{caseNum}")
    np.save(f"../data/revenues/{caseNum}/Revenue_q50.npy", R1)
    np.save(f"../data/revenues/{caseNum}/Revenue_ST.npy", R4)

    np.save(f"../data/revenues/{caseNum}/power_true.npy",modelling_table["Total_MWh"].values)
    np.save(f"../data/revenues/{caseNum}/power_pred.npy",modelling_table["mse"].values)
    np.save(f"../data/revenues/{caseNum}/pd_true.npy",modelling_table["Price_diff"].values)
    np.save(f"../data/revenues/{caseNum}/pd_pred.npy",modelling_table["pd_pred"].values)

methods_list=["ETS","arma","ar","persistance","persistance_average","persistance_seasonal","sarimax"]

for pathtype in ["test","latest"]:

    modelling_table=testTrading(pathtype=pathtype)
    modelling_table=calculateBidding(modelling_table,methods_list)
    showResults(modelling_table,methods_list,pathtype=pathtype)


    
    
