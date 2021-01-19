# setup git
git config --global user.name Archisman-Roy
git config --global user.email royarchi31@gmail.com

# pip install
pip install kaggle
pip3 install xgboost
pip install lightgbm
pip install ipywidgets
pip install optuna
pip install plotly # dependency in optuna module for visualization

# kaggle
export KAGGLE_USERNAME=royarchi
export KAGGLE_KEY=xxx

# download required data files
cd inputs
kaggle competitions download -c tabular-playground-series-jan-2021
unzip tabular-playground-series-jan-2021.zip
rm *.zip