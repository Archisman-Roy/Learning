# Use StratifiedKFold in a classification problems to maintain balances class distribution in each fold
# Use KFold in a regression problems

import pandas as pd
from sklearn import model_selection

problem_type = 'Regression'

if __name__ == "__main__":
    df = pd.read_csv("../inputs/train.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    if problem_type == 'Regression':
        kf = model_selection.KFold(n_splits=5, shuffle=False, random_state=None)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, 'kfold'] = fold
    
    elif problem_type == 'Classication':
        
        kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.target.values)):
            print(len(train_idx), len(val_idx))
            df.loc[val_idx, 'kfold'] = fold
    
    df.to_csv("../inputs/train_folds.csv", index=False)