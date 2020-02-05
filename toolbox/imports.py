#Tabular data manipulation and vectorization of loops
import numpy as np
import pandas as pd

#Pipe like Pandas
from dfply import *

#Multiprocess Pandas
import dask.dataframe as dd

#Time series extraction
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters

base_ts_parameters = {
 'sum_values': None,
 'absolute_sum_of_changes': None,
 'linear_trend': [{"attr": "slope"}],
 'skewness': None,
 'median': None,
 'mean': None}

#Visualization
import seaborn as sns

#Dataclass
from dataclasses import dataclass

#Type checking
from enum import IntEnum
from typing import Collection, Any
StrList = Collection[str]
