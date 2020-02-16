from .imports import *

#Describe a pandas dataframe
def advanced_describe(df):
    # get descriptive stats for dataframe for 'all' column dtypes
    desc = df.describe(include='all').T
    desc.drop(['top', 'freq', 'unique'], axis=1, inplace=True)
    
    # update column counts (df.describe() returns NaN for non-numeric cols)
    counts = pd.Series({ col: df[col].count() for col in df.columns })
    desc.update(counts.to_frame('count'))
    
    # add missing count/%
    missings = df.isnull().sum()
    desc = pd.concat([desc, missings.to_frame('missing')], axis=1, sort = True)
    desc['missing%'] = (desc['missing'] / len(desc)).round(2)

    # add unique counts/%
    uniques = pd.Series({ col: len(df[col].unique()) for col in df.columns })
    desc = pd.concat([desc, uniques.to_frame('unique')], axis=1, sort = True)
    desc['unique%'] = (desc['unique'] / len(desc)).round(2)
    
    unique_vals = pd.Series({ col: df[col].unique() for col in df.columns if len(df[col].unique()) < 20 })
    desc = pd.concat([desc, unique_vals.to_frame('unique_values')], axis=1, sort = True)
    
    # add col dtype
    dtypes = pd.Series({ col: df[col].dtype for col in df.columns })
    desc = pd.concat([desc, dtypes.to_frame('dtype')], axis=1, sort = True)
    
    return desc

def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)
            
            
def stringify_columns(df,column_list):
    """
    df: pandas dataframe
    column_list: list of columns which will be converted to a string
    
    Returns a copy of df with columns renamed
    """
    
    df_copy = df.copy()
    
    for i in column_list:
        df_copy[i] = df_copy[i].astype(str)
        
    return df_copy

def group_rolling_min(data, group_by_cols, time_col, shift_val, window_val, min_period_val, metric_col, metric_name):
    """
    data: Input dataframe
    group_by_cols: List of columns used to group by
    time_col: The columns against whom the data should be sorted, in most cases this will be datetime column
    shift_val: How much shifting is required for appropriate lead or lag
    window_val: rolling window
    min_period_val: Decides the minimum number of elements required in a window for computation
    metric_col: Column name where rolling will be applied
    metric_name: Name of output column
    """
    df = data.copy()
    
    #Sort
    df.sort_values(by=group_by_cols + [time_col])
    
    #Shift if needed
    df['shifted'] = df.groupby(group_by_cols)[metric_col].shift(shift_val)

    #Build reset list
    reset_list = []
    for i in range(len(group_by_cols)):
        reset_list.append(i)
    
    #Rolling
    df[metric_name] = df.groupby(group_by_cols)['shifted'].rolling(window=window_val,min_periods = min_period_val).min().reset_index(reset_list,drop=True)
    
    return df 