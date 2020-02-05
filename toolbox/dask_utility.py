from .imports import *


#Dask summarise operation for pandas df for quicker execution
def dd_summarise(data,group_by_cols,metric_dictionary, change_col_names = True, number_of_partitions = 32):
    """
    data: Pandas data-frame
    
    group_by_cols: List of columns to group_by on. It is safe to ensure there aren't any missing values in group_by cols
    
    metric_dictionary: This is a dictionary of fields and their summary function. 
    Example: metric_dictionary = {'col_1':['sum','min'],'col_2' : ['sum']}
    Dask functions can be found here: https://docs.dask.org/en/latest/dataframe-api.html#groupby-operations
    
    change_col_names: Defaults to True. This ensures columns are named appropriately like "col_name - summary_func"
    change_col_names as False will make columns named only as "col_name". 
    This could be an issue when there are multiple summary function with one column name in the metric dictionary.
    
    number_of_partitions: Defaults to 32. This should be based on number of cores in a machine
    Can run the following to check the number of cores: import multiprocessing; multiprocessing.cpu_count()
    
    This function returns the summarised dataframe
    """
    
    #Dask operation
    execution = dd.from_pandas(data, npartitions=number_of_partitions)
    execution = execution.groupby(group_by_cols).agg(metric_dictionary).compute()
    execution = execution.reset_index()
    
    #Column naming
    if change_col_names:
        execution.columns = [execution.columns.get_level_values(0)[x] 
         if execution.columns.get_level_values(1)[x] == '' 
         else execution.columns.get_level_values(0)[x] + "_" + execution.columns.get_level_values(1)[x]
         for x in range(len(execution.columns.get_level_values(1)))]
    else:
        execution.columns = execution.columns.get_level_values(0)
    return execution
