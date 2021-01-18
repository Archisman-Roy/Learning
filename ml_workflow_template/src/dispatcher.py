from sklearn import ensemble
import xgboost as xgb
from lightgbm import LGBMRegressor



lgb_params={'random_state': 33,
 'n_estimators':5000,
 'min_data_per_group': 5,
 'boosting_type': 'gbdt',
 'num_leaves': 256,
 'max_dept': -1,
 'learning_rate': 0.02,
 'subsample_for_bin': 200000,
 'lambda_l1': 1.074622455507616e-05,
 'lambda_l2': 2.0521330798729704e-06,
 'n_jobs': -1,
 'cat_smooth': 1.0,
 'silent': True,
 'importance_type': 'split',
 'metric': 'rmse',
 'feature_pre_filter': False,
 'bagging_fraction': 0.8206341150202605,
 'min_data_in_leaf': 100,
 'min_sum_hessian_in_leaf': 0.001,
 'bagging_freq': 6,
 'feature_fraction': 0.5,
 'min_gain_to_split': 0.0,
 'min_child_samples': 20}


xgb_params={'colsample_bytree' : 0.5,
            'alpha':0.01563,
            #'gamma':0.0,
            'learning_rate':0.01,
            'max_depth':15,
            'min_child_weight':257,
            'n_estimators':4000,                                                   
            #'reg_alpha':0.9,
            'reg_lambda':0.003,
            'subsample':0.7,
            'random_state':2020,
            'metric_period':100,
            'silent':1
}



MODELS = {
    "randomforestclassifier": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratreesclassifier": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "randomforestregressor": ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=2),
    "xgbregressor": xgb.XGBRegressor(**xgb_params),
    "lightgbmregressor":  LGBMRegressor(**lgb_params) 
}