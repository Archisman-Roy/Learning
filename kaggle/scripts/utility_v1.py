#This will contain all functions to handle V1 version of future sales prediction notebook
from .imports import *


def dd_summarise_m(data,group_by_cols,metric_dict):
    execution = dd.from_pandas(data, npartitions=32)
    execution = execution.groupby(group_by_cols).agg(metric_dict).compute()
    execution = execution.reset_index()
    execution.columns = [execution.columns.get_level_values(0)[x] 
     if execution.columns.get_level_values(1)[x] == '' 
     else execution.columns.get_level_values(0)[x] + "_" + execution.columns.get_level_values(1)[x]
     for x in range(len(execution.columns.get_level_values(1)))]
    return execution

def dd_summarise_m2(data,group_by_cols,metric_dict):
    execution = dd.from_pandas(data, npartitions=32)
    execution = execution.groupby(group_by_cols).agg(metric_dict).compute()
    execution = execution.reset_index()
    execution.columns = execution.columns.get_level_values(0)
    return execution


def apply_lag_v2(input_data,metric_list, number_of_lag_months):
    
    data = input_data #This avoids the functions to manipulate the input DF 
    
    for m in metric_list:
        for i in range(number_of_lag_months):
            data[f"{m}_lag_{i+1}"] = data.groupby(['item_id','shop_id'])[m].shift(i+1)
            print(f"lag: {i+1} done")
        print(f"Metric: {m} done")                                  
    return data


def apply_ts(data, group_by_cols ,columns_to_apply_ts, ts_params,summary_func, metric_name):
    
    input_data = data
    
    #Create metric dictionary
    md = {}
    for i in columns_to_apply_ts:
        md[i] = [summary_func]
        
    #Summarise
    summarised = dd_summarise_m2(input_data,group_by_cols,md)
    
    
    #Transpose
    summarised_t = summarised >> gather('month', metric_name, columns_to_apply_ts)
    summarised_t.sort_values(group_by_cols + ['month'],inplace = True)
    
    #Create composite key for feature extraction method
    temp = summarised_t
    composite = temp[group_by_cols] >> mutate(date_block_num = X.date_block_num.astype(str))
    composite['key'] = composite.values.sum(axis=1)
    summarised_t['key'] =  composite['key']
    temp = summarised_t[['key','month',metric_name]]
    
    #Extract features
    temp = extract_features(temp, column_id="key", column_sort="month", default_fc_parameters=ts_params)
    temp = temp.reset_index()
    
    #Create composite key for un transposed dataset
    composite = summarised[group_by_cols] >> mutate(date_block_num = X.date_block_num.astype(str))
    composite['id'] = composite.values.sum(axis=1)
    summarised['id'] =  composite['id']
    
    summarised = summarised >> left_join(temp, by = ['id']) >> drop(X.id)
    
    #Drop lag columns from summarised data
    summarised.drop(columns_to_apply_ts, axis=1,inplace = True)
    
    #Merge features
    output_data = input_data >> left_join(summarised, by = group_by_cols)
    
    return output_data

def item_presence_v1(data, columns_to_apply_ts):
    
    input_data = data
    
    #Transpose
    transposed = input_data >> gather('month', 'value', columns_to_apply_ts)
    transposed.sort_values(['item_id','shop_id','date_block_num','month'],inplace = True)
    
    transposed = transposed >> mask(X.value > 0) >> group_by(X.item_id, X.date_block_num) >> \
    summarise(item_presence = n_distinct(X.shop_id)) >> ungroup()
    
    output_data = input_data >> left_join(transposed, by = ['item_id','date_block_num']) >> \
    mutate(item_presence = if_else(X.item_presence.isnull(),0,X.item_presence))
           
    return output_data