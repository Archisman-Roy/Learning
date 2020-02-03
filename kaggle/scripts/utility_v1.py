#This will contain all functions to handle V1 version of future sales prediction notebook
from .imports import *


def dd_summarise_m(data,group_by_cols,metric_dict):
    execution = dd.from_pandas(data, npartitions=32)
    execution = execution.groupby(group_by_cols).agg(metric_dict).compute()
    execution = execution.reset_index()
    execution.columns = [execution.columns.get_level_values(0)[x] 
     if execution.columns.get_level_values(1)[x] == '' 
     else execution.columns.get_level_values(0)[x] + "_" + execution.columns.get_level_values(1) [x]
     for x in range(len(execution.columns.get_level_values(1)))]
    return execution


def apply_lag_v2(input_data,metric_list, number_of_lag_months):
    
    data = input_data #This avoids the functions to manipulate the input DF 
    
    for m in metric_list:
        for i in range(number_of_lag_months):
            data[f"{m}_lag_{i+1}"] = data.groupby(['item_id','shop_id'])[m].shift(i+1)
            print(f"lag: {i+1} done")
        print(f"Metric: {m} done")                                  
    return data