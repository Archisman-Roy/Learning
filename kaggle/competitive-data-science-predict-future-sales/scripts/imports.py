#Modelling library
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters

#Basic data manipualtion
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
from dfply import *
import dask.dataframe as dd


#Utility
import os
import zipfile
import math
import gc


#Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Others
from dataclasses import dataclass
from enum import IntEnum
from typing import Collection, Any #for type checking, explicit rules and evading python dynamic types
StrList = Collection[str]