# HEFTCom2024: Probabilistic Energy Forecasting and Trading Competition

![Static Badge](https://img.shields.io/badge/language-python-%20)
![Static Badge](https://img.shields.io/badge/license-MIT-a?color=blue)

This repository contains the code used in the paper **"A Hybrid Strategies for HEFTCom2024 Probabilistic Energy Forecasting and Trading Tasks based on Ensemble Learning Methods"**, which details the solutions developed by Team GEB for the [Hybrid Energy Forecasting and Trading Competition 2024 (HEFTCom2024)](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). Our solutions provide accurate probabilistic forecasts for a hybrid power plant and achieving significant trading revenue.

## Final Rank

In the final leaderboard of HEFTCom2024, the team GEB achieved:

- 🥇 1st place among student teams in the Combined Track.

- 🥉 3rd place overall in the Trading Track.

- 🎖 4th place overall in the Forecasting Track.

  |               | Trading Track | Forecasting Track | Combined Track |
  | ------------- | ------------- | ----------------- | -------------- |
  | Student Teams | 🥇1st         | 🥇1st             | 🥇1st          |
  | All           | 🥉3rd         | 🎖4th              | 🎖4th           |

## Overview

The codebase is structured to reproduce the key methods described in the paper:

1. **Stacking Models** trained on various Numerical Weather Predictions (NWPs) for wind power forecasting.
2. **Online Post-Processing** model to address distribution shifts caused by increased solar capacity in the online test set.
3. **Probabilistic Aggregation** technique to provide accurate quantile forecasts of total hybrid generation.
4. **Stochastic Trading Strategy** to maximize expected trading revenue considering uncertainties in electricity prices.
5. **Value-Oriented Price Spread Forecasting** to further enhance the trading revenue.

## Data Preparation

1. Download competition data from [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition).

2. Place raw data in `data/raw/` directory

## Data Preprocessing Pipeline

```bash
# Stage 1: Data normalization and feature engineering
python _01_dataPreProcess.py

# Stage 2: Dataset generation for historical/latest scenarios
python _02_generateDataset.py
python _03_generateLatestDataset.py
```

## Model Development

```bash
#Train LightGBM models for dense quantile regression
python _11_train.py

#Train the stacked multi-source NWPs sister forecasting model for wind power forecasting:
python _12_stacking_wind.py
```

## Hyperparameter Tuning

```
python params_search.py
```

## Experimental Validation

| Component                               | Validation Scripts                   | Metrics           |
| --------------------------------------- | ------------------------------------ | ----------------- |
| Wind Forecasting Ensemble               | `_21_test_wind_ensemble_*.py`        | Mean Pinball Loss |
| Solar Post-Processing                   | `_22_test_solar_online.py`           | Mean Pinball Loss |
| Probabilistic Aggregation               | `_23_test_aggregation_*.py`          | Mean Pinball Loss |
| Trading Strategy                        | `_24_test_trading_*.py`              | Trading Revenue   |
| Value-oriented Price Spread Forecasting | `_31_mse_oriented_*`,`_32_vof_\*.py` | Trading Revenue   |

## Analysis and Visualization

The following files are used to plot the figures involved in the paper:

| File Name               | Description                                                                                                                                                 |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `corelation.py`         | Analyzes the conditional correlation between wind and solar power generation.                                                                               |
| `plot_decision_loss.py` | Plots a heatmap showing the impact of power prediction errors and price spread prediction errors on trading revenue in the UK day-ahead electricity market. |
| `prices_anal.py`        | Analyzes the historical characteristics of the price spread within a day in the UK day-ahead electricity market.                                            |
| `solar_comp.py`         | Examines the impact of solar capacity growth in East England on solar power generation.                                                                     |

## Contact

For technical inquiries: Chuanqing Pu (sashabanks@sjtu.edu.cn)

## Acknowledgements

We would like to thank the organizers of HEFTCom2024 for providing the data and platform. We also thank Professor [Jethro Browell](https://github.com/jbrowell) and [Linwei Sang](https://github.com/sanglinwei) for their helpful suggestions on the research paper.
