from sklearn.model_selection import KFold
import numpy as np
#from comp_utils import pinball
import optuna
from lightgbm import LGBMRegressor
import pandas as pd

def objective(trial,X,y,quantile):
    
    params_grid={
        'num_leaves': trial.suggest_int('num_leaves', 100, 1000,step=100),
        "n_estimators": trial.suggest_categorical("n_estimators", [500,1000,2000]),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 200, 10000,step=100),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'lambda_l1': trial.suggest_int('lambda_l1', 0, 100,step=10),
        'lambda_l2': trial.suggest_int('lambda_l2', 0, 100,step=10),
        "random_state": 2048,
        'verbose':-1,
        'objective':'quantile  ',
        'alpha':quantile/100
        }

    cv = KFold(n_splits=5, shuffle=True, random_state=2048)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X,y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model=LGBMRegressor(**params_grid)
        model.fit(X_train,
                  y_train,
                  eval_set=[(X_test,y_test)],
                  callbacks=[optuna.integration.LightGBMPruningCallback(trial, "quantile")])
        
        preds=model.predict(X_test)
        
        cv_scores[idx] = np.mean((y_test-preds)**2)
        
    return np.mean(cv_scores)


if __name__=="__main__":
    
    #Load Dataset
    dataset=pd.read_csv("../data/dataset/full/WindDataset.csv")
    features=dataset.iloc[:,:-1]
    labels=dataset.iloc[:,-1]

    #Create study
    study = optuna.create_study(direction="minimize", study_name="Predictor_wind", storage="sqlite:///data/best_params/wind.db", load_if_exists=True)

    #Optimize
    func = lambda trial: objective(trial, features, labels, 0.5)
    study.optimize(func, n_trials=20)

    print(f"\tBest value (rmse): {study.best_value:.5f}")
    print("\tBest params_wind:")
    for key, value in study.best_params.items():
        print(f"\t\t{key}: {value}")

                

