from .imports import *

#DFPLY utility
def null_percent_original(series):
    return (pd.isnull(series).sum()/series.size) * 100

@make_symbolic
def null_percent(series):
    return null_percent_original(series)

@make_symbolic
def max_by(max_series, max_by_series):
    max_series = max_series.reset_index(drop = True)
    max_by_series = max_by_series.reset_index(drop = True)
    max_index = max_by_series.idxmax()
    return max_series[max_index]

@make_symbolic
def min_by(min_series, min_by_series):
    min_series = min_series.reset_index(drop = True)
    min_by_series = min_by_series.reset_index(drop = True)
    min_index = min_by_series.idxmin()
    return min_series[min_index]

@make_symbolic
def frank(series,ascending = True):
    ranks = series.rank(method='first', ascending=ascending)

