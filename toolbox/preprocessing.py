from .imports import *
from .utility import *


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
    def apply_train(self, df:pd.DataFrame):
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

    def apply_test(self, df:pd.DataFrame):
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
    def apply_train(self, df:pd.DataFrame):
        "Compute the means and stds of `self.cont_names` columns to normalize them."
        self.means,self.stds = {},{}
        for n in self.cont_names:
            assert is_numeric_dtype(df[n]), (f"""Cannot normalize '{n}' column as it isn't numerical.
                Are you sure it doesn't belong in the categorical set of columns?""")
            self.means[n],self.stds[n] = df[n].mean(),df[n].std()
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])

    def apply_test(self, df:pd.DataFrame):
        "Normalize `self.cont_names` with the same statistics as in `apply_train`."
        for n in self.cont_names:
            df[n] = (df[n]-self.means[n]) / (1e-7 + self.stds[n])
            


                
#Dates
def make_date(df:pd.DataFrame, date_field:str, custom_date_format:str):
    "Make sure `df[field_name]` is of the right date type."
    field_dtype = df[date_field].dtype
    if isinstance(field_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        field_dtype = np.datetime64
    if not np.issubdtype(field_dtype, np.datetime64):
        df[date_field] = pd.to_datetime(df[date_field], format = custom_date_format)

def add_datepart(df:pd.DataFrame, custom_date_format:str, field_name:str, prefix:str=None, drop:bool=True, time:bool=False):
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
