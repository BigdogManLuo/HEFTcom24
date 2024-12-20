# HEFTCom2024: Probabilistic Energy Forecasting and Trading Competition

![Static Badge](https://img.shields.io/badge/language-python-%20)
![Static Badge](https://img.shields.io/badge/license-MIT-a?color=blue)

This repository contains the code used in the paper **"A Hybrid Strategies for HEFTCom2024 Probabilistic Energy Forecasting and Trading Tasks based on Ensemble Learning Methods"**, which details the solutions developed by Team GEB for the [Hybrid Energy Forecasting and Trading Competition 2024 (HEFTCom2024)](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). In the final leaderboard, we ranked **1st** among student teams. We also ranked **3rd** in Trading Track and **4th** in Forecasting Track. Our solutions provide accurate probabilistic forecasts for a hybrid power plant and achieving significant trading revenue.

### Final Rank

|               | Trading Track | Forecasting Track | Combined Track |
| ------------- | ------------- | ----------------- | -------------- |
| Student Teams | 1st           | 1st               | 1st            |
| All           | 3rd           | 4th               | 4th            |

This code is primarily intended to reproduce all the experiments in our paper. For the engineering code used for online testing, please refer to the ["online"](https://github.com/BigdogManLuo/HEFTcom24/tree/online) branch.

## Overview

The codebase is structured to reproduce the key methods described in the paper:

1. **Stacking Models** trained on various Numerical Weather Predictions (NWPs) for wind power forecasting.
2. **Online Post-Processing** model to address distribution shifts caused by increased solar capacity in the online test set.
3. **Probabilistic Aggregation** technique to provide accurate quantile forecasts of total hybrid generation.
4. **Stochastic Trading Strategy** to maximize expected trading revenue considering uncertainties in electricity prices.
5. **Value-Oriented Price Spread Forecasting** to further enhance the trading revenue.

## Usage

### Data Preparation

The data used in this project is available on [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). Download the dataset and place it in the `./data/raw` directory.

### Data Preprocessing

To preprocess the dataset, run the following command:

```
python _01_dataPreProcess.py
```

To generate dataset for case1 and case2, run the following command:

```
python _02_generateDataset.py
python _03_generateLatestDataset.py
```

### Model Training

Train LightGBM models for dense quantile regression:

```
python _11_train.py
```

Train the stacked multi-source NWPs sister forecasting model for wind power forecasting:

```
python _12_stacking_wind.py
```

### Hyperparameter Tuning

run the following command:

```
python params_search.py
```

### Case Study

Validate the effectiveness of the stacked multi-source NWPs sister forecasting model:

```
python _21_test_wind_ensemble_history.py
python _21_test_wind_ensemble_latest.py
```

Validate the effectiveness of the solar online post-processing model:

```
python _22_test_solar_online.py
```

Validate the effectiveness of the probabilistic aggregation technique:

```
python _23_test_aggregation_history.py
python _23_test_aggregation_latest.py
```

Validate the effectiveness of the stochastic trading strategy:

```
python _24_test_trading_history.py
python _24_test_trading_latest.py
```

Validate the effectiveness of the value-oriented price spread forecasting:

```
python _31_vof_pre_train_history.py
python _32_vof_history.py
python _33_vof_pre_train_latest.py
python _34_vof_latest.py
```

### Others

The following files are used to plot the figures involved in the paper:

```
python corelation.py
python plot_decision_loss.py
python prices_anal.py
python solar_comp.py
```

## Acknowledgements

We would like to thank the organizers of HEFTCom2024 for providing the data and platform. We also thank Professor [Jethro Browell](https://github.com/jbrowell) and [Linwei Sang](https://github.com/sanglinwei) for their helpful suggestions on the research paper.
