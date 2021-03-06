import os
import pandas as pd
import xgboost as xgb
from sklearn import ensemble
from lightgbm import LGBMRegressor
from sklearn import preprocessing
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

PROBLEM_TYPE = os.environ.get("PROBLEM_TYPE")
MODEL = os.environ.get("MODEL")


def predict(test_data_path, model_type, model_path):
    
    df = pd.read_csv(test_data_path)
    test_idx = (df["id"].values)
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(test_data_path)
        df['target'] = 0
        df = df.drop(["id"], axis=1)
        
        # fetch continuous and categorical variables 
        cat_features = df.select_dtypes(exclude = 'number').columns.tolist()
        cont_features = df.select_dtypes(include = 'number').columns.tolist()
    
        # missing value treatment
        for col in cat_features:
            df[col].fillna('missing', inplace=True)
        for col in cont_features:
            df[col].fillna(-99999, inplace=True)
            

        # get encoder from training data
        encoding_pipeline = joblib.load(os.path.join(model_path, f"{MODEL}_{FOLD}_encoding_pipeline.pkl"))
        # get features
        cols = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}_columns.pkl"))
        # get model
        clf = joblib.load(os.path.join(model_path, f"{model_type}_{FOLD}.pkl"))
        
        df  = encoding_pipeline.transform(df) 
        df = df[cols]
        
        if PROBLEM_TYPE == 'Regression':
            preds = clf.predict(df)
        elif PROBLEM_TYPE == 'Classification':
            preds = clf.predict_proba(df)[:, 1] # probability of 1 when target classes are {0,1}

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])
    return sub
    

if __name__ == "__main__":
    submission = predict(test_data_path="inputs/test.csv", 
                         model_type=MODEL, 
                         model_path="models/")
    
    
    if PROBLEM_TYPE == 'Classification':
        pass # Convert class probabilities to class prediction in case of classification problem
    
    submission.id = submission.id.astype('int32') # validate the format in sample submission csv
    
    submission.to_csv(f"models/{MODEL}_submission.csv", index=False)