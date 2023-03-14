#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from IPython.display import Image
Image("../input/big-to-small-filepng/big_to_small_file.png")


# In[ ]:


# Some imports
import pandas as pd
DATA_PATH = "../input/elo-merchant-category-recommendation/historical_transactions.csv"


# In[ ]:


df = pd.read_csv(DATA_PATH)


# In[ ]:


get_ipython().run_cell_magic('timeit', '', 'df = pd.read_csv(DATA_PATH)')


# In[ ]:


ls -lh {DATA_PATH}


# In[ ]:


# verbose is set to False here to avoid the metadata information
df.info(verbose=False)


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


# Verbose is left to the default True here since we want the columns metadata.
df.select_dtypes('object').info(memory_usage='deep')


# In[ ]:


for col in df.select_dtypes('object'):
    print(df[col].sample(5))
    print(f"{df[col].nunique()} unique values for {col}, which has {len(df[col])} rows.")


# In[ ]:


df.purchase_date = pd.to_datetime(df.purchase_date)


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


CATEGORICAL_COLS = ["card_id", "category_3", "merchant_id"]
for col in["card_id", "category_3", "merchant_id"]:
    df[col] = df[col].astype("category")


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


for col in ["authorized_flag", "category_1"]:
    # Each row having "Y" (short for yes) will get the value 1, otherwise, 0.
    df[col] = pd.np.where(df[col] == "Y", 1, 0)


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


df.nunique().sort_values(ascending=True)


# In[ ]:


# Be careful, even though it is tempting to turn the "purchase_amount" to
# categorical to gain more space, 
# it isn't the best thing to do since we will be using this column to compute
# aggregations!
for col in ["month_lag", "installments", "state_id", "subsector_id", 
            "city_id", "merchant_category_id", "merchant_id"]:
    df[col] = df[col].astype("category")


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


df.dtypes


# In[ ]:


df.category_2.value_counts(dropna=False, normalize=True).plot(kind='bar')


# In[ ]:


df.category_2 = df.category_2.values.astype(int)


# In[ ]:


pd.__version__


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


# You can also use the "bool" type (both take one byte for storage).
df.authorized_flag = df.authorized_flag.astype(pd.np.uint8)
df.category_1 = df.category_1.astype(pd.np.uint8)


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


df.category_2 = df.category_2.astype(pd.np.uint8)


# In[ ]:


df.category_2.value_counts(normalize=True, dropna=False).plot(kind='bar')


# In[ ]:


df.info(memory_usage="deep", verbose=False)


# In[ ]:


# This function could be made generic to almost any loaded CSV file with
# pandas. Can you see how to do it?

# Some constants
PARQUET_ENGINE = "pyarrow"
DATE_COL = "purchase_date"
CATEGORICAL_COLS = ["card_id", "category_3", "merchant_id", "month_lag", 
                    "installments", "state_id", "subsector_id", 
                    "city_id", "merchant_category_id", "merchant_id"]
CATEGORICAL_DTYPES = {col: "category" for col in CATEGORICAL_COLS}
POSITIVE_LABEL = "Y"
INTEGER_WITH_NAN_COL = "category_2"
BINARY_COLS = ["authorized_flag", "category_1"]
INPUT_PATH = "../input/elo-merchant-category-recommendation/historical_transactions.csv"
OUTPUT_PATH = "historical_transactions.parquet"


def smaller_historical_transactions(input_path, output_path):
    # Load the CSV file, parse the datetime column and the categorical ones.
    df = pd.read_csv(input_path, parse_dates=[DATE_COL], 
                    dtype=CATEGORICAL_DTYPES)
    # Binarize some columns and cast to the boolean type
    for col in BINARY_COLS:
        df[col] = pd.np.where(df[col] == POSITIVE_LABEL, 1, 0).astype('bool')
    # Cast the category_2 to np.uint8
    df[INTEGER_WITH_NAN_COL] = df[INTEGER_WITH_NAN_COL].values.astype(pd.np.uint8)
    # Save as parquet file
    df.to_parquet(output_path, engine=PARQUET_ENGINE)
    return df
    
def load_historical_transactions(path=None):
    if path is None:
        return smaller_historical_transactions(INPUT_PATH, OUTPUT_PATH)
    else: 
        df = pd.read_parquet(path, engine=PARQUET_ENGINE)
        # Categorical columns aren't preserved when doing pandas.to_parquet
        # (or maybe I am missing something?)
        for col in CATEGORICAL_COLS:
            df[col] = df[col].astype('cateogry')
        return df


# In[ ]:


optimized_df = smaller_historical_transactions(INPUT_PATH, OUTPUT_PATH)


# In[ ]:


optimized_df.info(memory_usage="deep", verbose=False)


# In[ ]:


del df
del optimized_df


# In[ ]:


# TODO: There is a bug when reading the saved parquet file. Check why and fix it!
# Is it related to this issue: https://issues.apache.org/jira/browse/ARROW-2369?
# %%timeit 
# parquet_df = load_historical_transactions(INPUT_PATH)


# In[ ]:


# parquet_df.info(memory_usage="deep", verbose=False)


# In[ ]:


ls -lh {OUTPUT_PATH}

