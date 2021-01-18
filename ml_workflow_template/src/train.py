import os
import pandas as pd
import xgboost as xgb
from sklearn import ensemble
from sklearn import preprocessing
from lightgbm import LGBMRegressor

from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")

FOLD_MAPPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}

if __name__ == "__main__":
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # fetch continuous and categorical variables 
    cat_features = train_df.select_dtypes(exclude = 'number').columns.tolist()
    cont_features = train_df.select_dtypes(include = 'number').columns.tolist()
    
    if len(cat_features) > 0:
        pass # write code to encode categorical variables
        
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    
    if MODEL in ['xgbregressor']:
        clf.fit(train_df, ytrain,verbose=1,early_stopping_rounds=6, eval_set=[(valid_df, yvalid)])
    elif MODEL in ['lightgbmregressor']:
        clf.fit(train_df, ytrain,verbose=1,early_stopping_rounds=50, eval_set=((valid_df, yvalid)))
    elif MODEL in ['randomforestregressor']:
        clf.fit(train_df, ytrain)
    
    
    if PROBLEM_TYPE == 'Regression':
        preds = clf.predict(valid_df)
        print("R-squared: ", metrics.r2_score(yvalid, preds))
        print("RMSE: ", metrics.mean_squared_error(yvalid, preds,squared=False))
    elif PROBLEM_TYPE == 'Classification':
        preds = clf.predict_proba(valid_df)[:, 1] # probability of 1 when target classes are {0,1}
        print(metrics.roc_auc_score(yvalid, preds))

    joblib.dump(clf, f"models/{MODEL}_{FOLD}.pkl")
    joblib.dump(train_df.columns, f"models/{MODEL}_{FOLD}_columns.pkl")