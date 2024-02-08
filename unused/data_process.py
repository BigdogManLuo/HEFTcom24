import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.iolib.smpickle import load_pickle
import comp_utils

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle as pkl



#%% Weather Data
dwd_Hornsea1 = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_features=dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features=dwd_Hornsea1_features.merge(dwd_Hornsea1["WindDirection:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
dwd_Hornsea1_features=dwd_Hornsea1_features.merge(dwd_Hornsea1["Temperature"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index(),how="outer",on=["ref_datetime","valid_datetime"])
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")

#将dwd_Hornsea1_features中valid_datetime-ref_datetime大于6的数据删除
dwd_Hornsea1_features = dwd_Hornsea1_features[dwd_Hornsea1_features["valid_datetime"] - dwd_Hornsea1_features["ref_datetime"] < np.timedelta64(6,"h")]
