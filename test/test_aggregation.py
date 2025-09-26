import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
import utils_forecasting
import pandas as pd
import utils_data
import copy

def testAggregation(pathtype):
    
    # Load Data
    features_wind_dwd,labels_wind_dwd,features_solar_dwd,labels_solar_dwd=utils.loadFeaturesandLabels(pathtype=pathtype,source="dwd")
    features_wind_gfs,labels_wind_gfs,features_solar_gfs,labels_solar_gfs=utils.loadFeaturesandLabels(pathtype=pathtype,source="gfs")
    IntegratedDataset_test=pd.read_csv(f"../data/dataset/{pathtype}/IntegratedDataset.csv")
    
    if pathtype=="test":
        IntegratedDataset_test.drop(columns=["ref_datetime","valid_datetime","Wind_MWh_credit","Solar_MWh_credit","Wind_MWh_credit_y","Solar_MWh_credit_y","total_generation_MWh_y","DA_Price_y","SS_Price_y","DA_Price","SS_Price","hours_y"],inplace=True)
        labelsAllin1=IntegratedDataset_test["total_generation_MWh"]
        columns_wind_features_dwd,columns_solar_features_dwd=utils_data.getFeaturesName(source="dwd")
        columns_wind_features_gfs,columns_solar_features_gfs=utils_data.getFeaturesName(source="gfs")
        features_dwd=IntegratedDataset_test[columns_wind_features_dwd+columns_solar_features_dwd]
        features_gfs=IntegratedDataset_test[columns_wind_features_gfs+columns_solar_features_gfs]

        #Benchmark Model 
        Total_generation_forecast=utils_forecasting.forecastTotalByBenchmarkStacking(features_dwd,features_gfs)
        mpd_total_stacking=utils.meanPinballLoss(labelsAllin1,Total_generation_forecast)
        mws_total_stacking=utils.getWinklerScore(y_true=labelsAllin1,y_pred=Total_generation_forecast)
        mCRPS_stacking=utils.getMCRPS(y_true=labelsAllin1,y_pred=Total_generation_forecast)

    else:
        mpd_total_stacking,mws_total_stacking,mCRPS_stacking=0,0,0

    # Quantile Add
    if pathtype=="test":
        params_quantile_add={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "hours":features_solar_dwd[:,-1],
            "full":False,
            "WLimit":False,
            "SolarRevise":False,
            "rolling_test":False,
            "aggregation":False
        }

    elif pathtype=="latest":
        params_quantile_add={
            "wind_features_dwd":features_wind_dwd,
            "wind_features_gfs":features_wind_gfs,
            "solar_features":features_solar_dwd,
            "hours":features_solar_dwd[:,-1],
            "full":True,
            "WLimit":True,
            "SolarRevise":True,
            "rolling_test":True,
            "availableCapacities":IntegratedDataset_test["availableCapacity"].values,
            "aggregation":False
        }

    params_aggregation=copy.deepcopy(params_quantile_add)
    params_aggregation["aggregation"]=True

    Total_generation_forecast0,_,_=utils_forecasting.forecast_total(**params_quantile_add)
    mpd_total0=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast0)
    mws_total0=utils.getWinklerScore(y_true=labels_wind_dwd+labels_solar_dwd,y_pred=Total_generation_forecast0)
    mCRPS0=utils.getMCRPS(y_true=labels_wind_dwd+labels_solar_dwd,y_pred=Total_generation_forecast0)

    Total_generation_forecast1,_,_=utils_forecasting.forecast_total(**params_aggregation)
    mpd_total1=utils.meanPinballLoss(labels_wind_dwd+labels_solar_dwd,Total_generation_forecast1)
    mws_total1=utils.getWinklerScore(y_true=labels_wind_dwd+labels_solar_dwd,y_pred=Total_generation_forecast1)
    mCRPS1=utils.getMCRPS(y_true=labels_wind_dwd+labels_solar_dwd,y_pred=Total_generation_forecast1)

    print(f"----------------------------Case {pathtype} -----------------------------")

    print("==================Mean Pinball Loss==================")
    print(f"Quantile by Quantile:{mpd_total0}\nWith Aggregation:{mpd_total1}",f"\nTotal Forecasting:{mpd_total_stacking}")

    print("==================Mean Continuous Ranked Probability Score==================")
    print(f"Quantile by Quantile:{mCRPS0}\nWith Aggregation:{mCRPS1}",f"\nTotal Forecasting:{mCRPS_stacking}")

    print("==================Winkler Score==================")
    print(f"Quantile by Quantile:{mws_total0}\nWith Aggregation:{mws_total1}",f"\nTotal Forecasting:{mws_total_stacking}")

    return Total_generation_forecast0,Total_generation_forecast1,labels_wind_dwd+labels_solar_dwd

if __name__ == "__main__":
    _,_,_=testAggregation(pathtype="test")
    Total_generation_forecast0,Total_generation_forecast1,labels_total=testAggregation(pathtype="latest")
    utils.plotPowerGeneration(Total_generation_forecast0,labels_total,filename="total_gen_quantile_add.png",x_range0=2400,ptype="total")
    utils.plotPowerGeneration(Total_generation_forecast1,labels_total,filename="total_gen_aggregation.png",x_range0=2400,ptype="total")

