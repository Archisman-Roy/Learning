from .imports import *


def item_category_size(base,items):
    cat_size = items >> group_by(X.item_category_id) >> summarise(cat_size = n_distinct(X.item_id)) >> ungroup()
    cat_size = items >> left_join(cat_size, by = ['item_category_id']) >> select(X.item_id,X.cat_size,X.item_category_id) >> \
    mutate(item_id = X.item_id.astype(str))
    
    base = base >> \
    left_join(cat_size, by = ['item_id']) >> \
    mutate(cat_size=if_else(X.cat_size.isnull(),-999,X.cat_size),
           item_category_id=if_else(X.item_category_id.isnull(),-999,X.item_category_id))
 
    return base

def my_plot_importance(booster, figsize, **kwargs):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return xgb.plot_importance(booster=booster, ax=ax, **kwargs)

def rmse(x,y): return math.sqrt(((x-y)**2).mean())