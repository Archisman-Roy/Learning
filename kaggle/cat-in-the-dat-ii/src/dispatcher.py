from sklearn import ensemble
import xgboost as xgb
from lightgbm import LGBMRegressor,LGBMClassifier
from . import params


MODELS = {
    "randomforestclassifier": ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "extratreesclassifier": ensemble.ExtraTreesClassifier(n_estimators=200, n_jobs=-1, verbose=2),
    "randomforestregressor": ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=2),
    "xgbregressor": xgb.XGBRegressor(**params.PARAMS['xgb']),
    "lightgbmregressor":  LGBMRegressor(**params.PARAMS['lgb']),
    "hptlightgbmregressor":  LGBMRegressor(**params.PARAMS['lgb_hpt']),
    "lightgbmclassifier": LGBMClassifier(**params.PARAMS['lgbc'])
}