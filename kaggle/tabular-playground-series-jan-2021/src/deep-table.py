if __name__ == "__main__":
        
    # imports
    from deeptables.models.deeptable import DeepTable, ModelConfig
    from deeptables.datasets import dsutils
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import joblib
    
    # get data
    data = pd.read_csv("inputs/train_folds.csv")

    # train and valid inputs
    y = data.pop('target')
    X = data.drop(['id','kfold'],axis=1)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # model config
    conf = ModelConfig(
        metrics=['RootMeanSquaredError'], 
        nets=['dnn_nets','dcn_nets'],
        dnn_params={
            'hidden_units': ((512, 0.3, True), (512, 0.3, True)),
            'dnn_activation': 'relu',
        },
        earlystopping_patience=5,
    )

    # set config
    dt = DeepTable(config=conf)

    # train
    model, history = dt.fit(X_train, y_train, epochs=100)

    # evaluate
    score = dt.evaluate(X_test, y_test)
    
    joblib.dump(score, f"models/dt_score.pkl")
    joblib.dump(model, f"models/dt_model.pkl")
    joblib.dump(dt, f"models/dt_dt.pkl")
