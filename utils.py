import pickle
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from itertools import chain
from scipy.interpolate import interp1d
from sklearn.linear_model import QuantileRegressor
from sklearn.utils.fixes import parse_version, sp_version
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import scienceplots


def loadFeaturesandLabels(pathtype,source):

    WindDataset=pd.read_csv(f"../data/dataset/{pathtype}/{source}/WindDataset.csv")
    SolarDataset=pd.read_csv(f"../data/dataset/{pathtype}/{source}/SolarDataset.csv")

    features_wind=WindDataset.iloc[:,:-1].values
    labels_wind=WindDataset.iloc[:,-1].values

    features_solar=SolarDataset.iloc[:,:-1].values
    labels_solar=SolarDataset.iloc[:,-1].values

    return features_wind,labels_wind,features_solar,labels_solar


#============================Evaluation=================================
def pinball(y,y_hat,alpha):

    return ((y-y_hat)*alpha*(y>=y_hat) + (y_hat-y)*(1-alpha)*(y<y_hat)).mean()

def getRevenue(Xb,Xa,yd=None,ys=None):
    return yd*Xb+(Xa-Xb)*ys-0.07*(Xa-Xb)**2

def getEquivalentPriceSpreadForecast(Xb,Xa):
    price_diff=(Xb-Xa)*0.14
    return price_diff

def getPinballLosses(y_true,y_pred):
    pinball_losses={}
    for i in range(10,100,10):
        pinball_losses[f"q{i}"]=pinball(y_true,y_pred[f"q{i}"],alpha=i/100)
    return pinball_losses

def meanPinballLoss(y_true, y_pred):
    mpd=0
    for i in range(10,100,10):
        mpd+=pinball(y_true,y_pred[f"q{i}"],alpha=i/100)
    return mpd/9

def crps(x_value, F_x, x_obs):
    """
    Calculate the Continuous Ranked Probability Score (CRPS).
    :param x_value: ndarray, range of values for integration
    :param F_x: ndarray, Cumulative Distribution Function (CDF) values corresponding to x_value
    :param x_obs: float, the observed value
    :return: float, CRPS value
    """
    # Calculate 𝟙(y ≥ x_obs)
    step_function = np.where(x_value >= x_obs, 1.0, 0.0)
    
    # Compute the CRPS using numerical integration (trapezoidal rule)
    crps_value = np.trapz((F_x - step_function) ** 2, x_value)
    
    return crps_value

def getMCRPS(y_true,y_pred):
    CRPSs=[]
    quantiles = y_pred.keys()
    forecast_array = np.array([y_pred[q] for q in quantiles])
    if forecast_array.shape[0]==101:
        quantiles = np.arange(0,101,1)
    elif forecast_array.shape[0]==9:
        quantiles = np.arange(10,100,10)
    for i in range(forecast_array.shape[1]):
        probabilities = quantiles / 100.0
        cdf=interp1d(forecast_array[:,i],probabilities,kind='linear')
        x_values = np.arange(forecast_array[:,i].min(), forecast_array[:,i].max(), 1e-1)
        cdf_values = cdf(x_values)
        CRPSs.append(crps(x_values, cdf_values, y_true[i]))

    return np.mean(CRPSs)

def winkler_score(y_pred, y_true, alpha=0.2):
    
    if y_pred.shape[0]==101:
        lower_idx=int((100*alpha)/2)
        upper_idx=int(100-(100*alpha)/2)
    elif y_pred.shape[0]==9:
        lower_idx=0
        upper_idx=-1
    lower = y_pred[lower_idx]
    upper = y_pred[upper_idx] 

    width = upper - lower
    penalty_lower = np.where(y_true < lower, (2 / alpha) * (lower - y_true), 0)
    penalty_upper = np.where(y_true > upper, (2 / alpha) * (y_true - upper), 0)

    winkler_scores = width + penalty_lower + penalty_upper
    
    return winkler_scores

def getWinklerScore(y_true,y_pred):
    winkler_scores=[]
    quantiles = y_pred.keys()
    forecast_array = np.array([y_pred[q] for q in quantiles]) 
    for i in range(forecast_array.shape[1]):
        winkler_scores.append(winkler_score(forecast_array[:,i], y_true[i]))

    return np.mean(winkler_scores)

def getCoverageProbability(y_true,y_pred):
    coverage_probabilities=[]
    quantiles = y_pred.keys()
    forecast_array = np.array([y_pred[q] for q in quantiles])
    coverage_probabilities=np.mean((forecast_array[-1,:] > y_true) & (forecast_array[0,:] < y_true))
    return coverage_probabilities

def plotPowerGeneration(Generation_forecast,labels,filename, x_range0,step_size=590,ptype="solar"):
    
    time_series=pd.date_range(start="2024-02-19 23:00",end="2024-05-19 22:30",freq="30T",tz="UTC")
    
    x_range1=x_range0+step_size
    x_range=np.arange(x_range0,x_range1)
    plt.style.use('science')
    plt.figure(figsize=(12,7))
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams.update({'font.size': 25})
    l1=plt.plot(time_series[x_range],labels[x_range],label="true",linewidth=2,color="b")
    colors=["#cf6a87","#f8a5c2"]
    l2=plt.fill_between(time_series[x_range],Generation_forecast["q10"][x_range],Generation_forecast["q30"][x_range],color=colors[0],alpha=0.5)
    l3=plt.fill_between(time_series[x_range],Generation_forecast["q30"][x_range],Generation_forecast["q50"][x_range],color=colors[1],alpha=0.5)
    plt.fill_between(time_series[x_range],Generation_forecast["q50"][x_range],Generation_forecast["q70"][x_range],color=colors[1],alpha=0.5)
    plt.fill_between(time_series[x_range],Generation_forecast["q70"][x_range],Generation_forecast["q90"][x_range],color=colors[0],alpha=0.5)

    plt.legend([l1[0],l2,l3],["True","q10-q30, q70-q90","q30-q70"],frameon=False,bbox_to_anchor=(1, 1.15),ncol=3)
    xticks_labels = [ts.strftime("%m-%d") for ts in time_series[x_range]]
    xticks_indices = np.arange(23, len(time_series[x_range]),96)  # 96个半小时间隔为两天
    xticks_labels = [time_series[x_range][i].strftime("%m-%d") for i in xticks_indices]
    plt.xticks(ticks=time_series[x_range][xticks_indices], labels=xticks_labels)
    
    if ptype=="total":
        plt.yticks(np.arange(0,1400,step=200))
    
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.tick_params(width=2)

    plt.xlabel("Date")
    if ptype=="solar":
        plt.ylabel("Solar Power(MWh)")
    elif ptype=="total":
        plt.ylabel("Total Power(MWh)")

    plt.savefig(f"../figs/{filename}",dpi=660)
    




