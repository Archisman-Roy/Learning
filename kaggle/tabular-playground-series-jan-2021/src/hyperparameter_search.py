import os
import pandas as pd
import xgboost as xgb
from sklearn import ensemble
from sklearn import preprocessing
from lightgbm import LGBMRegressor
import optuna
from optuna import Trial
from sklearn.metrics import mean_squared_error
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")

def fit(trial, xtr, ytr, xval, yval):
    
    params = {'random_state'      : 33,
         'n_estimators'      : trial.suggest_categorical("n_estimators", [500, 1000, 2000, 4000, 5000, 6000]),
         'min_data_per_group': trial.suggest_categorical("min_data_per_group", [2, 5, 10, 20, 40, 80]),
         'boosting_type'     : 'gbdt',
         'num_leaves'        : trial.suggest_categorical("num_leaves", [64, 128, 256, 512]),
         'max_depth'                 : -1,
         'learning_rate'            : trial.suggest_loguniform("learning_rate",0.005,0.1),
         'subsample_for_bin'        : 200000,
         'lambda_l1'                : 1.074622455507616e-05,
         'lambda_l2'                : 2.0521330798729704e-06,
         'n_jobs'                   : -1,
         'cat_smooth'               : 1.0,
         'silent'                   : True,
         'importance_type'          : 'split',
         'metric'                   : 'rmse',
         'feature_pre_filter'       : False,
         'bagging_fraction'         : 0.8206341150202605,
         'min_data_in_leaf'         : trial.suggest_categorical("min_data_in_leaf", [10, 50, 100, 200, 400]),
         'min_sum_hessian_in_leaf'  : 0.001,
         'bagging_freq'             : 6,
         'feature_fraction'         : 0.5,
         'min_gain_to_split'        : 0.0,
         'min_child_samples'        : 20}
    
    model = LGBMRegressor(**params)
    model.fit(xtr, ytr,verbose=1,early_stopping_rounds=50, eval_set=((xval, yval)))
    
    y_val_pred = model.predict(xval)
    
    log = {
        "train rmse": mean_squared_error(ytr, model.predict(xtr), squared=False), # setting squared=False returns root_mean_squared_error
        "valid rmse": mean_squared_error(yval, y_val_pred, squared=False)  # setting squared=False returns root_mean_squared_error
    }

    return log

def objective(trial):
    rmse = 0
    for fold in range(5):
        df = pd.read_csv(TRAINING_DATA)
        train_df = df[df.kfold!= fold].reset_index(drop=True)
        valid_df = df[df.kfold==fold].reset_index(drop=True)

        ytrain = train_df.target.values
        yvalid = valid_df.target.values

        train_df = train_df.drop(["id", "target", "kfold"], axis=1)
        valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

        valid_df = valid_df[train_df.columns]
        
        log = fit(trial, train_df, ytrain, valid_df, yvalid)
        rmse += log['valid rmse']/5
        
    return rmse

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name='LGBM optimization')
    study.optimize(objective, n_trials=300)

    joblib.dump(study, f"models/hyperparametertuning_study.pkl")