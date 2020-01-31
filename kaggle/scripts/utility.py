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

#String funtions

def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]

def mid(s, offset, amount):
    return s[offset:offset+amount]

def remove_from_right(s, amount):
    return s[:-amount]


#Others
def ifnone(a:Any,b:Any)->Any:
    "`a` if `a` is not None, otherwise `b`."
    return b if a is None else a


#data pre-processing

FillStrategy = IntEnum('FillStrategy', 'MEDIAN COMMON CONSTANT')

@dataclass
class TabularProc():
    "A processor for tabular dataframes."
    cat_names:StrList
    cont_names:StrList

    def __call__(self, df:pd.DataFrame, test:bool=False):
        "Apply the correct function to `df` depending on `test`."
        func = self.apply_test if test else self.apply_train
        func(df)

    def apply_train(self, df:pd.DataFrame):
        "Function applied to `df` if it's the train set."
        pass
    def apply_test(self, df:pd.DataFrame):
        "Function applied to `df` if it's the test set."
        pass


class Categorify(TabularProc):
    "Transform the categorical variables to that type."
    def apply_train(self, df:pd.DataFrame):
        "Transform `self.cat_names` columns in categorical."
        self.categories = {}
        for n in self.cat_names:
            df.loc[:,n] = df.loc[:,n].astype('category').cat.as_ordered()
            self.categories[n] = df[n].cat.categories

    def apply_test(self, df:pd.DataFrame):
        "Transform `self.cat_names` columns in categorical using the codes decided in `apply_train`."
        for n in self.cat_names:
            df.loc[:,n] = pd.Categorical(df[n], categories=self.categories[n], ordered=True)
            

@dataclass
class category_to_num(TabularProc):
    max_n_cat:int=None
    def apply_train(self, df:pd.DataFrame):
        self.col_unique_count = {}
        for n in self.cat_names:
            self.col_unique_count[n] = df[n].nunique()
            if (self.max_n_cat is None or self.col_unique_count[n]>self.max_n_cat): df.loc[:,n] = df[n].cat.codes+1
            else: pd.get_dummies(df, columns=[n], dummy_na = False)

    def apply_test(self, df:pd.DataFrame):
        for n in self.cat_names:
            if (self.max_n_cat is None or self.col_unique_count[n] > self.max_n_cat): df.loc[:,n] = df[n].cat.codes+1
            else: pd.get_dummies(df, columns=[n], dummy_na = False)

                
@dataclass
class FillMissing(TabularProc):
    "Fill the missing values in continuous columns."
    fill_strategy:FillStrategy= FillStrategy.MEDIAN
    add_col:bool=True
    fill_val:float=0.
    def apply_train(self, df:DataFrame):
        "Fill missing values in `self.cont_names` according to `self.fill_strategy`."
        self.na_dict = {}
        for name in self.cont_names:
            if pd.isnull(df[name]).sum():
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                if self.fill_strategy == FillStrategy.MEDIAN: filler = df[name].median()
                elif self.fill_strategy == FillStrategy.CONSTANT: filler = self.fill_val
                else: filler = df[name].dropna().value_counts().idxmax()
                df[name] = df[name].fillna(filler)
                self.na_dict[name] = filler

    def apply_test(self, df:DataFrame):
        "Fill missing values in `self.cont_names` like in `apply_train`."
        for name in self.cont_names:
            if name in self.na_dict:
                if self.add_col:
                    df[name+'_na'] = pd.isnull(df[name])
                    if name+'_na' not in self.cat_names: self.cat_names.append(name+'_na')
                df[name] = df[name].fillna(self.na_dict[name])
            elif pd.isnull(df[name]).sum() != 0:
                print(f"""There are nan values in field {name} but there were none in the training set. 
                Please fix those manually.""")

class Normalize(TabularProc):
    "Normalize the continuous variables."
    def apply_train(self, df:DataFrame):
        "Compute the means and stds of `self.cont_names` columns to normalize them."
        self.means,self.stds = {},{}
        for n in self.cont_names:
            assert is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical.
                Are you sure it doesn't belong in the categorical set of columns?""")
            self.means[n],self.stds[n] = df[n].mean(),df[n].std()
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df:DataFrame):
        "Normalize `self.cont_names` with the same statistics as in `apply_train`."
        for n in self.cont_names:
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])
            


                
#Dates
def make_date(df:DataFrame, date_field:str, custom_date_format:str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], format = custom_date_format)

def add_datepart(df:DataFrame, custom_date_format:str, field_name:str, prefix:str=None, drop:bool=True, time:bool=False):
    "Helper function that adds columns relevant to a date in the column `field_name` of `df`."
    df["date_original"] = df[field_name]
    make_date(df, field_name,custom_date_format)
    field = df[field_name]
    prefix = ifnone(prefix, re.sub('[Dd]ate$', '', field_name))
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start', 
            'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[prefix + n] = getattr(field.dt, n.lower())
    df[prefix + 'Elapsed'] = field.astype(np.int64) // 10 ** 9
    if drop: df.drop(field_name, axis=1, inplace=True)
    return df

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
    return ranks


#Dask functions

def dd_summarise(data,group_by_cols,col_name, summarise_func):
    execution = dd.from_pandas(data, npartitions=32)
    execution = execution.groupby(group_by_cols).agg({col_name: [summarise_func]}).compute()
    execution = execution.reset_index()
    execution.columns = execution.columns.get_level_values(0)
    return execution

#Model data

cl = ['item_id','shop_id']

def add_labels(data,base):
    y = data >> mask(X.date_flag == 'y')
    execution = dd.from_pandas(y, npartitions=32)
    y = execution.groupby(cl).item_cnt_day.sum().compute()
    y = (y.to_frame()).reset_index()
    
    base  = base >> left_join(y, by = cl) >> mutate(item_cnt_day = if_else(X.item_cnt_day.isnull(), 0 , X.item_cnt_day))
    
    return base


def create_model_data(data):
    
    base = data >> select(X.shop_id,X.item_id) >> distinct(X.shop_id,X.item_id)
    base = add_labels(data,base)
    
    base = base >> mutate(composite = X.item_id + X.shop_id)
    
    print(base.shape)
    
    return base


def create_sparse_data(data,train,composite_columns_list):

    month = data >> distinct(X.date_block_num) >> select(X.date_block_num) >> mutate(key = 1)
    composite_key = train[composite_columns_list]
    composite_key = composite_key.drop_duplicates()
    composite_key = composite_key >> mutate(key = 1)
    ts = composite_key >> inner_join(month,by = ['key']) >> drop(X.key)
    
    data_x = data >> mask(X.date_flag == 'x')
    
    execution = dd.from_pandas(data_x, npartitions=32)
    actuals = execution.groupby(composite_columns_list + ["date_block_num"]).item_cnt_day.sum().compute()
    actuals = (actuals.to_frame()).reset_index()
    
    ts = ts >> left_join(actuals, by = composite_columns_list + ["date_block_num"] ) >> \
    mutate(item_cnt_day = if_else(X.item_cnt_day.isnull(),0,X.item_cnt_day))
    return ts

def ts_features(data,params):
    
    temp = data
    composite = temp >> drop(X.item_cnt_day,X.date_block_num)
  
    composite['key'] = composite.values.sum(axis=1)
    temp['key'] =  composite['key']
    temp = temp >> select(X.key,X.date_block_num,X.item_cnt_day)

    extracted_features = extract_features(temp, column_id="key", column_sort="date_block_num", default_fc_parameters=params)
    extracted_features = extracted_features.reset_index()
    extracted_features = extracted_features.rename(columns={"id": "key"}, inplace = False)
    return extracted_features


def apply_ts_features(data,ts_data,composite_column):
    
    temp = data.rename(columns={composite_column: "key"}, inplace = False)
    ts_data.columns = [composite_column + '_' + x if x != 'key' else x for x in ts_data.columns.to_list()]
    
    #Hacky solve
    if composite_column == 'composite':
        data_final = temp >> left_join(ts_data, by = ['key']) >> drop(X.key)
    else:
        data_final = temp >> left_join(ts_data, by = ['key'])
        data_final.rename(columns={"key":composite_column}, inplace = True)
    
    return data_final


def remove_elements(li, el):
    for i in el:
        li.remove(i)
    return li


def add_prediction_month_feature(base,data,prediction_month):
    
    data_x = data >> mask(X.date_flag == 'x',X.Month == prediction_month) 
    
    execution = dd.from_pandas(data_x, npartitions=32)
    feature = execution.groupby(cl).item_cnt_day.mean().compute()
    feature = (feature.to_frame()).reset_index()
    feature = feature >> rename(pred_month_avg_composite = X.item_cnt_day)
    base = base >> \
    left_join(feature, by = cl) >> \
    mutate(pred_month_avg_composite=if_else(X.pred_month_avg_composite.isnull(),0,X.pred_month_avg_composite))
    
    execution = dd.from_pandas(data_x, npartitions=32)
    feature = execution.groupby(['item_id']).item_cnt_day.mean().compute()
    feature = (feature.to_frame()).reset_index()
    feature = feature >> rename(pred_month_avg_item = X.item_cnt_day)
    base = base >> \
    left_join(feature, by = ['item_id']) >> \
    mutate(pred_month_avg_item=if_else(X.pred_month_avg_item.isnull(),0,X.pred_month_avg_item))
    
    execution = dd.from_pandas(data_x, npartitions=32)
    feature = execution.groupby(['shop_id']).item_cnt_day.mean().compute()
    feature = (feature.to_frame()).reset_index()
    feature = feature >> rename(pred_month_avg_shop = X.item_cnt_day)
    base = base >> \
    left_join(feature, by = ['shop_id']) >> \
    mutate(pred_month_avg_shop=if_else(X.pred_month_avg_shop.isnull(),0,X.pred_month_avg_shop))
     
    return base


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m,x_train,y_train,x_valid,y_valid):
    res = [rmse(m.predict(x_train), y_train.values.ravel()), rmse(m.predict(x_valid), y_valid.values.ravel()),
                m.score(x_train, y_train), m.score(x_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)