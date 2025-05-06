# HEFTCom2024: Probabilistic Energy Forecasting and Trading Competition

![Static Badge](https://img.shields.io/badge/language-python-%20)
![Static Badge](https://img.shields.io/badge/license-MIT-a?color=blue)

This repository contains the code used in the paper **"A Hybrid Strategies for HEFTCom2024 Probabilistic Energy Forecasting and Trading Tasks based on Ensemble Learning Methods"**, which details the solutions developed by Team GEB for the [Hybrid Energy Forecasting and Trading Competition 2024 (HEFTCom2024)](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition). Our solutions provide accurate probabilistic forecasts for a hybrid power plant and achieving significant trading revenue.
![image](figs/p1.png)

## Final Rank

In the final leaderboard of HEFTCom2024, the team GEB achieved:

- 🥇 1st place among student teams in the Combined Track.

- 🥉 3rd place overall in the Trading Track.

- 🎖 4th place overall in the Forecasting Track.

![image](figs/p2.png)

## Overview

The codebase is structured to reproduce the key methods described in the paper:

1. **Stacking Sister Models** trained on various Numerical Weather Predictions (NWPs) for wind power forecasting.
2. **Online Post-Processing** model to address distribution shifts caused by increased solar capacity in the online test set.
3. **Probabilistic Aggregation** technique to provide accurate quantile forecasts of total hybrid generation.
4. **Stochastic Trading Strategy** to maximize expected trading revenue considering uncertainties in electricity prices.
5. **End-to-End Learning in Trading** to further enhance the trading revenue.

## Data Preparation

1. Download competition data from [IEEE Dataport](https://ieee-dataport.org/competitions/hybrid-energy-forecasting-and-trading-competition).

2. Place raw data in `data/raw/` directory

## Data Preprocessing Pipeline

```bash
# Data normalization, feature engineering,  dataset generation for historical/latest scenarios
python pre-process/dataPreProcess.py
```

## Model Development

```bash
#Train LightGBM models for dense quantile regression
python train/train.py

#Train total generation forecasting model
python train/train_total_allin1.py
python train/train_total_stacking.py

#Train models that DWD and GFS features are both embedded
python train/train_allin1.py

#Train the stacked multi-source NWPs sister forecasting model
python stacking.py
```

## Hyperparameter Tuning

```
python hyper-params_search/params_search.py
```

## Experimental Validation

| Component                 | Validation Scripts                                          | Metrics                           |
| ------------------------- | ----------------------------------------------------------- | --------------------------------- |
| Wind Forecasting Ensemble | `test/test_ensemble.py`                                     | Pinball Loss, CRPS, Winkler Score |
| Solar Post-Processing     | `test/test_solar_online.py`                                 | Pinball Loss, CRPS, Winkler Score |
| Probabilistic Aggregation | `test/test_aggregation.py`                                  | Pinball Loss, CRPS, Winkler Score |
| Trading Strategy          | `test/test_trading.py`                                      | Trading Revenue                   |
| End-to-End Learning       | `test/test_e2e_benchmark_case*.py`,`test/test_e2e_case*.py` | Trading Revenue                   |

## Analysis and Visualization

The following files are used to plot the figures involved in the paper:

| File Name                           | Description                                                                                                                                                 |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data_analys/corelation.py`         | Analyzes the conditional correlation between wind and solar power generation.                                                                               |
| `data_analys/plot_decision_loss.py` | Plots a heatmap showing the impact of power prediction errors and price spread prediction errors on trading revenue in the UK day-ahead electricity market. |
| `data_analys/prices_anal.py`        | Analyzes the historical characteristics of the price spread within a day in the UK day-ahead electricity market.                                            |
| `data_analys/solar_comp.py`         | Examines the impact of solar capacity growth in East England on solar power generation.                                                                     |
| `data_analys/draw_NWP_map.py`       | Plotting the coordinates of NWP data from DWD and GFS provided by HEFTcom24 in Hornsea 1 and the East England PV plant area                                 |
| `data_analys/plot_revenue.py`       | Plotting a scatter plot of the power prediction error and price spread prediction error of different methods on the trading track                           |

## Contact

For technical inquiries: Chuanqing Pu (sashabanks@sjtu.edu.cn)

## Acknowledgements

We would like to thank the organizers of HEFTCom2024 for providing the data and platform. We also thank Professor [Jethro Browell](https://github.com/jbrowell) and [Linwei Sang](https://github.com/sanglinwei) for their helpful suggestions on the research paper.
