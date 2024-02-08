import comp_utils
import pandas as pd
import numpy as np
import xarray as xr
from statsmodels.iolib.smpickle import load_pickle
from sklearn.decomposition import PCA

dwd_Hornsea1 = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
#dwd_Hornsea1_features=dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
#dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
#dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")


dwd_solar = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")

def getPrincipalFeature(feature_name):
    
    global dwd_Hornsea1

    latitude=[53.77, 53.84, 53.9, 53.97, 54.03, 54.1]
    longitude=[1.702, 1.767, 1.832, 1.897, 1.962, 2.027]

    dwd_Hornsea1_features={}
    for la in latitude:
        for lo in longitude:
            dwd_Hornsea1_features[(la,lo)] = dwd_Hornsea1[feature_name].sel(latitude=la,longitude=lo,method="nearest").to_dataframe()
            #处理缺失值，用缺失值数据前后的均值填充
            dwd_Hornsea1_features[(la,lo)] = dwd_Hornsea1_features[(la,lo)].fillna(method='ffill')
            dwd_Hornsea1_features[(la,lo)] = dwd_Hornsea1_features[(la,lo)][feature_name]

    features = np.array(list(dwd_Hornsea1_features.values()))
    pca = PCA(n_components=1)
    features = pca.fit_transform(features.T)
    
    # 查看每个主成分的方差解释率
    print(pca.explained_variance_ratio_)

    return features



def getPrincipalSolarFeature(feature_name="SolarDownwardRadiation",n_components=1):
    
    global dwd_solar
    
    points=np.arange(0,20)
    
    dwd_solar_features={}

    for point in points:
        dwd_solar_features[point]=dwd_solar[feature_name].sel(point=point,method="nearest").to_dataframe()
        #处理缺失值，用缺失值数据前后的均值填充
        dwd_solar_features[point]=dwd_solar_features[point].fillna(method='ffill')
        dwd_solar_features[point]=dwd_solar_features[point][feature_name]


    features = np.array(list(dwd_solar_features.values()))
    pca = PCA(n_components=n_components)
    features = pca.fit_transform(features.T)
    
    # 查看每个主成分的方差解释率
    print(pca.explained_variance_ratio_)

    return features





features_rad=getPrincipalSolarFeature(n_components=2)







features_ws_100=getPrincipalFeature("WindSpeed:100")
features_wd_100=getPrincipalFeature("WindDirection:100")
features_temperature=getPrincipalFeature("Temperature")

#按列合并
features = np.concatenate((features_ws_100, features_wd_100, features_temperature), axis=1)


dwd_solar = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")













