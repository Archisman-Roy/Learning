from .imports import *

#Sensitive info
for line in open("kaggle.txt","r").readlines(): # Read the lines
        login_info = line.split() 

#Set parameters
os.environ['KAGGLE_USERNAME'] = login_info[1]
os.environ['KAGGLE_KEY'] = login_info[2]
 
#! pip install kaggle
from kaggle.api.kaggle_api_extended import KaggleApi 

def download_data(competition_name,current_dir):
    api = KaggleApi(login_info[0])
    api.authenticate()
    api.competition_download_files(competition_name)
    with zipfile.ZipFile(current_dir + '/' + competition_name + '.zip', 'r') as zip_ref:
        zip_ref.extractall(current_dir + '/' + 'data')


def read_data(path, file_format, file_name):
    
    if file_format == "csv":
        return pd.read_csv(f"{path}{file_name}.{file_format}")