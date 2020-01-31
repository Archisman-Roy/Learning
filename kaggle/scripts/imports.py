#Modelling library
from sklearn.ensemble import RandomForestRegressor
from tsfresh import extract_features

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

#Visualization
import seaborn as sns

#Others
from dataclasses import dataclass
from enum import IntEnum
from typing import Collection, Any #for type checking, explicit rules and evading python dynamic types
StrList = Collection[str]



