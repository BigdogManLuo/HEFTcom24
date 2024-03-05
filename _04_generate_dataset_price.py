import pandas as pd
import numpy as np
import xarray as xr

#加载原始数据
demand_old=xr.open_dataset("data/dwd_icon_eu_demand_20200920_20231027.nc")
demand_old_gfs=xr.open_dataset("data/ncep_gfs_demand_20200920_20231027.nc")
