import os
import pandas as pd
import xgboost as xgb
from sklearn import ensemble
from sklearn import preprocessing
from lightgbm import LGBMRegressor
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
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
    train_df = df[df.kfold.isin(FOLD_MAPPPING.get(FOLD))].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    train_df = train_df.drop(["id", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "kfold"], axis=1)

    valid_df = valid_df[train_df.columns]

    # fetch continuous and categorical variables 
    cat_features = train_df.select_dtypes(exclude = 'number').columns.tolist()
    cont_features = train_df.select_dtypes(include = 'number').columns.tolist()
    
    # missing value treatment
    for col in cat_features:
        train_df[col].fillna('missing', inplace=True)
        valid_df[col].fillna('missing', inplace=True)
    for col in cont_features:
        train_df[col].fillna(-99999, inplace=True)
        valid_df[col].fillna(-99999, inplace=True)
    
    # cat encoding
    ohe_list = ['bin_3', 'bin_4', 'nom_0', 'nom_4']
    te_list =  ['nom_1', 'nom_2', 'nom_3', 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
     
    # run target encoding and one hot encoding using pipeline (for simplicity and avoiding temp table creations)
    encoding_pipeline = Pipeline([
      ('ohe', ce.OneHotEncoder(cols=ohe_list, use_cat_names=True, handle_unknown = 'error', return_df=True)),
      ('te', ce.TargetEncoder(cols=te_list, smoothing=0, handle_unknown = 'value', return_df=True))
    ])
    # Get the encoded training set:
    train_df = encoding_pipeline.fit_transform(train_df, train_df['target'])

    # Get the encoded valid set
    valid_df  = encoding_pipeline.transform(valid_df)  
    
    train_df = train_df.drop(["target"], axis=1)
    valid_df = valid_df.drop(["target"], axis=1)
        
    # data is ready to train
    clf = dispatcher.MODELS[MODEL]
    
    if MODEL in ['xgbregressor']:
        clf.fit(train_df, ytrain,verbose=1,early_stopping_rounds=6, eval_set=[(valid_df, yvalid)])
    elif MODEL in ['lightgbmregressor','hptlightgbmregressor']:
        clf.fit(train_df, ytrain,verbose=1,early_stopping_rounds=50, eval_set=((valid_df, yvalid)))
    elif MODEL in ['randomforestregressor','randomforestclassifier']:
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
    joblib.dump(encoding_pipeline, f"models/{MODEL}_{FOLD}_encoding_pipeline.pkl")