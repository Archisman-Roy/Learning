from .imports import *
from .pandas_utility import *
from .dask_utility import *


def apply_lag(df, group_by_cols_list, time_unit_col, metric_list, lag_list):
    """
    This function returns lag variables for numeric fields at a particular group_by level. 
    
    df: Pandas data frame
    group_by_cols_list: the list of fields to group by
    time_unit_col: This field tells this function how to correctly sort the data by time so as to obtain correct lags
    metric_list: list of numeric fields for which lag needs to be computed
    lag_list: List of lag time periods, example: [1,2,3,6,12] 
    """
    df_copy = df.copy()
    df_copy.sort_values(group_by_cols_list + [time_unit_col],inplace = True)
    for m in metric_list:
        for i in lag_list:
            df_copy[f"{m}_lag_{i}"] = df_copy.groupby(group_by_cols_list)[m].shift(i)
            print(f"lag: {i} done")
        print(f"Metric: {m} done")                                  
    return df_copy



def apply_ts(df, group_by_cols ,columns_to_apply_ts, ts_params, summary_func, metric_name,  time_unit, drop_lag_fields = True):
    """
    This function returns time series features from time lag columns in data
    You can also specify the aggregation levels in group_by_cols to define the required granualrity
    
    df: Input pandas dataframe
    group_by_cols: This is the key which has all the lag columns associated with it
    columns_to_apply_ts: the columns which forms the time series
    ts_params: time series metrics, passing this as None will ignore the time series extraction section
    summary_func: This is the summary function applied when we different level of data from the level present in df
    metric_name: This will be used as prefix to all new time features
    time_unit: The name of time field which gets generated when the lag fields are transposed to a single column
    drop_lag_fields: Flag to specify if lag fields will be dropped, if they are not dropped then metric_name is used as prefix in the lag fields
    """
    df_copy = df.copy()
    
    #Create metric dictionary
    md = {}
    for i in columns_to_apply_ts:
        md[i] = [summary_func]
        
    #Summarise
    summarised = dd_summarise(df_copy,group_by_cols,md, change_col_names = False)
    
    
    if ts_params is not None:
        #Transpose
        summarised_t = summarised >> gather(time_unit, metric_name, columns_to_apply_ts)
        summarised_t.sort_values(group_by_cols + [time_unit],inplace = True)
    
        #Create composite key for feature extraction method
        composite = summarised_t[group_by_cols]
        composite = stringify_columns(composite,group_by_cols) #Change all group_by_cols to str so that concatenations works
        composite['key'] = composite.values.sum(axis=1)
        summarised_t['key'] =  composite['key']
        summarised_t = summarised_t[['key',time_unit,metric_name]]
    
        #Extract features
        temp = extract_features(summarised_t, column_id="key", column_sort=time_unit, default_fc_parameters=ts_params)
        temp = temp.reset_index()
    
        #Create composite key for un transposed dataset
        composite = summarised[group_by_cols]
        composite = stringify_columns(composite,group_by_cols) #Change all group_by_cols to str so that concatenations works
        composite['id'] = composite.values.sum(axis=1)
        summarised['id'] =  composite['id']
    
        summarised = summarised >> left_join(temp, by = ['id']) >> drop(X.id)
    
    #Drop lag columns from summarised data
    if drop_lag_fields:
        summarised.drop(columns_to_apply_ts, axis=1,inplace = True)
    else:
        for i in columns_to_apply_ts:
            summarised.rename(columns={ i : metric_name + '_' + i }, inplace = True)
            
    #Merge features
    output_data = df_copy >> left_join(summarised, by = group_by_cols)
    
    return output_data