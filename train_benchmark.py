import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.formula.api as smf
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#%% Weather Data
dwd_Hornsea1 = xr.open_dataset("data/dwd_icon_eu_hornsea_1_20200920_20231027.nc")
dwd_Hornsea1_features=dwd_Hornsea1["WindSpeed:100"].mean(dim=["latitude","longitude"]).to_dataframe().reset_index()
dwd_Hornsea1_features["ref_datetime"] = dwd_Hornsea1_features["ref_datetime"].dt.tz_localize("UTC")
dwd_Hornsea1_features["valid_datetime"] = dwd_Hornsea1_features["ref_datetime"] + pd.TimedeltaIndex(dwd_Hornsea1_features["valid_datetime"],unit="hours")


dwd_solar = xr.open_dataset("data/dwd_icon_eu_pes10_20200920_20231027.nc")
dwd_solar_features = dwd_solar["SolarDownwardRadiation"].mean(dim="point").to_dataframe().reset_index()
dwd_solar_features["ref_datetime"] = dwd_solar_features["ref_datetime"].dt.tz_localize("UTC")
dwd_solar_features["valid_datetime"] = dwd_solar_features["ref_datetime"] + pd.TimedeltaIndex(dwd_solar_features["valid_datetime"],unit="hours")

#%% Energy Data
energy_data = pd.read_csv("data/Energy_Data_20200920_20231027.csv")
energy_data["dtm"] = pd.to_datetime(energy_data["dtm"])
energy_data["Wind_MWh_credit"] = 0.5*energy_data["Wind_MW"] - energy_data["boa_MWh"]
energy_data["Solar_MWh_credit"] = 0.5*energy_data["Solar_MW"]

#%% Merge Energy and Weather Data
modelling_table = dwd_Hornsea1_features.merge(dwd_solar_features,how="outer",on=["ref_datetime","valid_datetime"])
modelling_table = modelling_table.set_index("valid_datetime").groupby("ref_datetime").resample("30T").interpolate("linear")
modelling_table = modelling_table.drop(columns="ref_datetime",axis=1).reset_index()
modelling_table = modelling_table.merge(energy_data,how="inner",left_on="valid_datetime",right_on="dtm")
modelling_table = modelling_table[modelling_table["valid_datetime"] - modelling_table["ref_datetime"] < np.timedelta64(50,"h")]
modelling_table.rename(columns={"WindSpeed:100":"WindSpeed"},inplace=True)

#%% Modelling
modelling_table = modelling_table[modelling_table["SolarDownwardRadiation"].notnull()]
modelling_table = modelling_table[modelling_table["WindSpeed"].notnull()]
modelling_table["total_generation_MWh"] = modelling_table["Wind_MWh_credit"] + modelling_table["Solar_MWh_credit"]

mod_wind = smf.quantreg('Wind_MWh_credit ~ bs(WindSpeed,df=8)',data=modelling_table)
mod_solar= smf.quantreg('Solar_MWh_credit ~ bs(SolarDownwardRadiation,df=5)',data=modelling_table)
Models_wind = dict()
Models_solar = dict()

for quantile in range(10,100,10):
    Models_wind[f"q{quantile}"] = mod_wind.fit(q=quantile/100,max_iter=2500)
    Models_solar[f"q{quantile}"] = mod_solar.fit(q=quantile/100,max_iter=2500)

    modelling_table[f"q{quantile}"] = Models_wind[f"q{quantile}"].predict(modelling_table)+Models_solar[f"q{quantile}"].predict(modelling_table)
    modelling_table.loc[modelling_table[f"q{quantile}"] < 0, f"q{quantile}"] = 0


#%% Save Models
for quantile in range(10,100,10):
    Models_wind[f"q{quantile}"].save(f"models/benchmark/wind_q{quantile}.pickle")
    Models_solar[f"q{quantile}"].save(f"models/benchmark/solar_q{quantile}.pickle")
