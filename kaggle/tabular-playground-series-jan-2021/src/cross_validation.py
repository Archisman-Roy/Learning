import pandas as pd
from sklearn import model_selection
import math

"""
- -- binary classification
- -- multi class classification
- -- multi label classification
- -- single column regression
- -- multi column regression
- -- holdout
"""


class CrossValidation:
    def __init__(
            self,
            df, 
            target_cols,
            shuffle, 
            problem_type="single_col_regression",
            multilabel_delimiter=",",
            stratifiedKfold_in_regression = True, 
            num_folds=5,
            random_state=42
        ):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle,
        self.random_state = random_state
        self.multilabel_delimiter = multilabel_delimiter
        self.stratifiedKfold_in_regression = stratifiedKfold_in_regression

        if self.shuffle is True:
            self.dataframe = self.dataframe.sample(frac=1).reset_index(drop=True)
        
        self.dataframe["kfold"] = -1
    
    def split(self):
        if self.problem_type in ("binary_classification", "multiclass_classification"):
            if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.dataframe[target].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            elif unique_values > 1:
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe[target].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold

        elif self.problem_type in ("single_col_regression", "multi_col_regression"):
            if self.num_targets != 1 and self.problem_type == "single_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets < 2 and self.problem_type == "multi_col_regression":
                raise Exception("Invalid number of targets for this problem type")
            
            if self.problem_type in ("single_col_regression") and self.stratifiedKfold_in_regression == True:
                # bin_count = 1 + int(math.log(self.dataframe.shape[0],2)) # Sturges' rule 
                bin_count = 4
                self.dataframe['bins'] = pd.qcut(self.dataframe.target, bin_count, labels=False)
                
                kf = model_selection.StratifiedKFold(n_splits=self.num_folds, 
                                                     shuffle=False)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe, y=self.dataframe['bins'].values)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
    
                self.dataframe.drop(['bins'], axis=1, inplace = True)
            else:
                kf = model_selection.KFold(n_splits=self.num_folds)
                for fold, (train_idx, val_idx) in enumerate(kf.split(X=self.dataframe)):
                    self.dataframe.loc[val_idx, 'kfold'] = fold
        
        elif self.problem_type.startswith("holdout_"):
            pass # code later

        elif self.problem_type == "multilabel_classification":
            pass # code later

        else:
            raise Exception("Problem type not understood!")

        return self.dataframe


if __name__ == "__main__":
    df = pd.read_csv("../inputs/train.csv")
    cv = CrossValidation(df, shuffle=True, target_cols=["target"])
    df_split = cv.split()
    df_split.to_csv('../inputs/train_folds.csv',index = False)
    print(df_split.head())
    print(df_split.kfold.value_counts())