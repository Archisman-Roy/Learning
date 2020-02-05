from .imports import *

def item_presence(data, columns_to_apply_ts):
    
    input_data = data.copy()
    
    #Transpose
    transposed = input_data >> gather('month', 'value', columns_to_apply_ts)
    transposed.sort_values(['item_id','shop_id','date_block_num','month'],inplace = True)
    
    transposed = transposed >> mask(X.value > 0) >> group_by(X.item_id, X.date_block_num) >> \
    summarise(item_presence = n_distinct(X.shop_id)) >> ungroup()
    
    output_data = input_data >> left_join(transposed, by = ['item_id','date_block_num']) >> \
    mutate(item_presence = if_else(X.item_presence.isnull(),0,X.item_presence))
           
    return output_data


def item_category_size(base,items):
    cat_size = items >> group_by(X.item_category_id) >> summarise(cat_size = n_distinct(X.item_id)) >> ungroup()
    cat_size = items >> left_join(cat_size, by = ['item_category_id']) >> select(X.item_id,X.cat_size,X.item_category_id) >> \
    mutate(item_id = X.item_id.astype(str))
    
    base = base >> \
    left_join(cat_size, by = ['item_id']) >> \
    mutate(cat_size=if_else(X.cat_size.isnull(),-999,X.cat_size),
           item_category_id=if_else(X.item_category_id.isnull(),-999,X.item_category_id))
 
    return base