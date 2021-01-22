# setup git
git config --global user.name Archisman-Roy
git config --global user.email royarchi31@gmail.com

# module installation
pip install kaggle
pip3 install xgboost
pip install lightgbm
pip install ipywidgets
pip install optuna
pip install plotly # dependency in optuna module for visualization
# python -m pip install featuretools # autamated feature engineering
# python -m pip install "featuretools[complete]" # add-ons like tsfresh, categorical encoding etc
# pip install pipenv # Good to have, but not necessary
pip install torch torchvision # fastai dependency
pip install fastai
pip install ipython-autotime



# kaggle
export KAGGLE_USERNAME=royarchi
export KAGGLE_KEY=xxx

# download required data files
cd inputs
kaggle competitions download -c cat-in-the-dat-ii
unzip cat-in-the-dat-ii.zip
rm *.zip