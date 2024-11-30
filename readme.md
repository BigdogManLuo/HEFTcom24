# HEFTCom2024: Probabilistic Energy Forecasting and Trading Competition

![Static Badge](https://img.shields.io/badge/language-python-%20)
![Static Badge](https://img.shields.io/badge/license-MIT-a?color=blue)

This repository contains the code developed by team GEB for the [Hybrid Energy Forecasting and Trading Competition 2024 (HEFTCom2024)](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). In the final leaderboard, we ranked **1st** among student teams. We also ranked **3rd** in Trading Track and **4th** in Forecasting Track. Our solutions provide accurate probabilistic forecasts for a hybrid power plant and achieving significant trading revenue.

### Final Rank

|               | Trading Track | Forecasting Track | Combined Track |
| ------------- | ------------- | ----------------- | -------------- |
| Student Teams | 1st           | 1st               | 1st            |
| All           | 3rd           | 4th               | 4th            |

## Usage

### Data Preparation

The data used in this project is available on [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). Download the dataset and place it in the `./data/raw` directory.

API documentation can be found on the rebase.energy website for energy data and weather data, and some basic wrappers are included in `comp_utils.py`.

### Data Preprocessing

To preprocess the raw data, run the following command:

```
python  _01_dataPreProcess.py
```

To generate dataset for training, run the following command:

```
python  _02_generateDataset.py
```

To merge the latest data for trading, run the following command:

```
python  _03_merge_latest_dataset_dwd.py
python  _03_merge_latest_dataset_gfs.py
```

### Model Training

Train LightGBM models for dense quantile regression:

```
python  _11_train.py
```

Train the stacked multi-source sister forecasting model for wind power forecasting:

```
python _12_stacking_wind.py
```

Train the solar online post-processing model for solar power forecasting:

```
python _13_solar_revised.py
```

Train the solar online post-processing model for solar power forecasting (for bidding use):

```
python _14_solar_revised_bidding.py
```

### Offline Evaluation

```
python _21_test_wind_ensemble_history.py
python _22_test_trading_history.py
```

### Online Submission

`auto_submiter.py` accesses the latest weather forecast data through the team's API Key, generates the forecast results, submits them to the server, and records the submitted results locally.

### Retrospective analysis

`comparision.py` is applied in the online testing stage. By comparing the submitted results with the real energy production data and market data, the prediction accuracy and revenue are calculated to evaluate the performance of the model.

## Acknowledgements

We would like to thank the organizers of HEFTCom2024 for providing the data and platform.
