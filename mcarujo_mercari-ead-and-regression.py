#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install nb_black -q')


# In[2]:


get_ipython().run_line_magic('load_ext', 'nb_black')


# In[3]:


get_ipython().run_cell_magic('bash', '', '\napt install --assume-yes p7zip-full\n7z x ../input/mercari-price-suggestion-challenge/train.tsv.7z -y\n7z x ../input/mercari-price-suggestion-challenge/test.tsv.7z -y\n7z x ../input/mercari-price-suggestion-challenge/test_stg2.tsv.zip -y\n7z x ../input/mercari-price-suggestion-challenge/sample_submission.csv.7z -y\n7z x ../input/mercari-price-suggestion-challenge/sample_submission_stg2.csv.zip -y')


# In[4]:


import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from string import punctuation
from string import punctuation
import nltk

warnings.filterwarnings("ignore")


# In[5]:


train = pd.read_csv(
    "train.tsv",
    sep="\t",
    usecols=[
        "name",
        "item_condition_id",
        "category_name",
        "brand_name",
        "price",
        "shipping",
        "item_description",
    ],
)
train.head()


# In[6]:


train.brand_name.fillna("No Brand", inplace=True)
train.dropna(inplace=True)
train.info()


# In[7]:


punctuation = [p for p in punctuation]
stopwords = nltk.corpus.stopwords.words("english")
stopwords = stopwords + punctuation + ["..."] + ["!!"]
token_punct = nltk.WordPunctTokenizer()
stemmer = nltk.RSLPStemmer()


# In[8]:


def plot_value_counts(serie, name_column, number=20):
    y = serie.value_counts()[:number].values
    x = serie.value_counts()[:number].index

    fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, textposition="auto",)])
    fig.update_layout(
        title_text="Counting " + name_column,
        xaxis_title=name_column,
        yaxis_title="count",
    )
    fig.show()


def hist_plot(serie, titles=["Histogram", "Acumulative"]):
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles)
    fig.add_trace(
        go.Histogram(x=serie), row=1, col=1,
    )
    fig.add_trace(
        go.Histogram(x=serie, cumulative_enabled=True), row=1, col=2,
    )
    fig.show()


def remove_punct(my_str):
    no_punct = ""
    for char in my_str:
        if char not in punctuation:
            no_punct = no_punct + char
    return no_punct


def tokenizer_column(serie):
    clear_col = list()
    for row in serie:
        new_line = list()
        line = token_punct.tokenize(remove_punct(row.lower()))
        for word in line:
            if word not in stopwords:  # stopwords
                new_line.append(stemmer.stem(word))
        clear_col.append(" ".join(new_line))
    return clear_col


def wordcloud(text, column_name, title):
    all_words = " ".join([text for text in text[column_name]])
    wordcloud = WordCloud(
        width=800, height=500, max_font_size=110, collocations=False
    ).generate(all_words)
    plt.figure(figsize=(24, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.show()


# In[9]:


column = train.category_name
print("How much NAN we have here?", column.isna().sum())
print("How much categories we have here?", len(column.unique()))
train.category_name = train.category_name.fillna("no category")


# In[10]:


plot_value_counts(column, "category_name")


# In[11]:


def transform_split_category_name(df):
    aux = df["category_name"].str.split("/", n=2, expand=True)
    for i in [0, 1, 2]:
        df["category_name_" + str(i)] = aux[i]
        df["category_name_" + str(i)].fillna("No category", inplace=True)

    return df


# In[12]:


transform_split_category_name(train)
plot_value_counts(train.category_name_0, "category_name_0")
plot_value_counts(train.category_name_1, "category_name_1")
plot_value_counts(train.category_name_2, "category_name_2")


# In[13]:


train = train[train.category_name_0.isin(["Electronics"])]


# In[14]:


plot_value_counts(train.category_name_1, "category_name_1")


# In[15]:


plot_value_counts(train.category_name_2, "category_name_2")


# In[16]:


train.info()


# In[17]:


column = train.name
print("How much NAN we have here?", column.isna().sum())
print("How much categories we have here?", len(column.unique()))


# In[18]:


train.name = tokenizer_column(train.name)


# In[19]:


plot_value_counts(column, "name")


# In[20]:


wordcloud(train, "name", "Name wordcloud")


# In[21]:


train.item_description = tokenizer_column(train.item_description)


# In[22]:


wordcloud(train, "item_description", "Description wordcloud")


# In[23]:


column = train.item_condition_id
print("How much NAN we have here?", column.isna().sum())
print("How much categories we have here?", len(column.unique()))
plot_value_counts(column, "item_condition_id")


# In[24]:


column = train.brand_name
print("How much NAN we have here?", column.isna().sum())
print("How much categories we have here?", len(column.unique()))


# In[25]:


train.brand_name = train.brand_name.fillna("no brand")
train.brand_name = train.brand_name.str.lower()


# In[26]:


plot_value_counts(column.sample(10000), "brand_name")


# In[27]:


list_top_50_brand = train.brand_name.value_counts()[:51].index
list_top_50_brand


# In[28]:


brands_column = []
for i, brand in enumerate(train.brand_name):
    if not brand in list_top_50_brand:
        brands_column.append("other brand")
    else:
        brands_column.append(brand)
brands_column[:5]


# In[29]:


train.brand_name = brands_column
plot_value_counts(column.sample(10000), "brand_name", 52)


# In[30]:


column = train.price
print("How much NAN we have here?", column.isna().sum())


# In[31]:


train = train[train.price.isin([0]) == False]  # removing 0 prices
roof = train.price.quantile(0.98)  # removing outlines
print(f"removing values higher than {roof}")
train = train.query(f"price < {roof}")


# In[32]:


hist_plot(column.sample(10000))


# In[33]:


train["log_price"] = np.log(train.price)
hist_plot(train["log_price"].sample(10000))


# In[34]:


column = train.shipping
print("How much NAN we have here?", column.isna().sum())
print("How much categories we have here?", len(column.unique()))


# In[35]:


plot_value_counts(column.sample(10000), "shipping")


# In[36]:


train.info()


# In[37]:


px.box(train.sample(10000), x="item_condition_id", y="price", color='item_condition_id',title="Price boxplot by condition id")


# In[38]:


table = train.groupby("item_condition_id")["price"].describe().round(3)
ff.create_table(table, height_constant=40, index=True, index_title="item_condition_id")


# In[39]:


# 1 means free ship for the customer
# 0 means the customer have to pay the ship

table = train.groupby("item_condition_id")["shipping"].describe()[["count", "mean"]]
table["free ship in %"] = table["mean"].round(3) * 100
table["not free ship in %"] = 100 - (table["mean"].round(3) * 100)
ff.create_table(
    table.round(3), height_constant=40, index=True, index_title="item_condition_id"
)


# In[40]:


most_popular_categories = train.category_name.value_counts()[:30].index


def count_cond_cat(id_v):
    aux = train[
        train.item_condition_id.isin([id_v])
        & train.category_name.isin(most_popular_categories)
    ]
    aux = aux.category_name_0.value_counts()
    x = aux.index
    y = aux.values
    return x, y


# In[41]:


fig = go.Figure()

x, y = count_cond_cat(1)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 1"))

x, y = count_cond_cat(2)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 2"))

x, y = count_cond_cat(3)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 3"))

x, y = count_cond_cat(4)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 4"))

x, y = count_cond_cat(5)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 5"))

fig.update_layout(
    title="Comparative item_condition by most popular categories in our dataset",
    xaxis_tickfont_size=14,
    yaxis=dict(title="count", titlefont_size=16, tickfont_size=14,),
    legend=dict(
        bgcolor="rgba(255, 255, 255, 0)", bordercolor="rgba(255, 255, 255, 0)",
    ),
    barmode="group",
    bargap=0.15,  # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,  # gap between bars of the same location coordinate.
)
fig.show()


# In[42]:


most_popular_categories = train.brand_name.value_counts()[
    :30
].index  # skipping the most popular: 'No brand' hahaha


def count_brand_cat(id_v):
    aux = train[
        train.item_condition_id.isin([id_v])
        & train.brand_name.isin(most_popular_categories)
    ]
    aux = aux.brand_name.value_counts()
    x = aux.index
    y = aux.values
    return x, y


# In[43]:


fig = go.Figure()

x, y = count_brand_cat(1)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 1"))

x, y = count_brand_cat(2)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 2"))

x, y = count_brand_cat(3)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 3"))

x, y = count_brand_cat(4)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 4"))

x, y = count_brand_cat(5)
fig.add_trace(go.Bar(x=x, y=y, name="item_condition_id = 5"))

fig.update_layout(
    title="Comparative brand_name by most popular categories in our dataset",
    xaxis_tickfont_size=14,
    yaxis=dict(title="count", titlefont_size=16, tickfont_size=14,),
    legend=dict(
        bgcolor="rgba(255, 255, 255, 0)", bordercolor="rgba(255, 255, 255, 0)",
    ),
    barmode="group",
    bargap=0.15,  # gap between bars of adjacent location coordinates.
    bargroupgap=0.1,  # gap between bars of the same location coordinate.
)
fig.show()


# In[44]:


roof = train.price.quantile(0.95)
px.box(
    train.query(f"price < {roof}").sample(10000),
    x="brand_name",
    y="price",
    color="brand_name",
    title="Price boxplot by brand_name",
)


# In[45]:


px.box(
    train.sample(10000),
    x="brand_name",
    y="log_price",
    color="brand_name",
    title="Price boxplot by brand_name with log transformation",
)


# In[46]:


train = train.sample(10000)
train.reset_index(inplace=True, drop=True)
train.head()


# In[47]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import (
    BayesianRidge,
    SGDRegressor,
)

# split the dataset beteween train and test
def split(x, y, plot=False):
    # seed
    # train_test_split
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.1, random_state=42367,
    )
    if plot:
        print(
            "sizes: train (x,y) and test (x,y)",
            train_x.shape,
            train_y.shape,
            test_x.shape,
            test_y.shape,
        )
    return train_x, test_x, train_y, test_y


# Just train and valid the model
def run_reg_linear(train_x, test_x, train_y, test_y, model, plot=False):
    model.fit(train_x, train_y)
    test_pred = model.predict(test_x)

    mse = mean_squared_error(test_y, test_pred)
    mae = mean_absolute_error(test_y, test_pred)
    r2 = r2_score(test_y, test_pred)

    if plot:
        print("*" * 40)
        print("r2 score", r2)
        print("mse", mse)
        print("mae", mae)
        print("*" * 40)

    return r2, mse


# Train with all models then return a table with scores
def train_test_show(train_x, test_x, train_y, test_y):
    valores = []
    models = [
        ("BayesianRidge", BayesianRidge()),
        ("MLPRegressor", MLPRegressor()),
        ("SGDRegressor", SGDRegressor()),
        ("RandomForestRegressor", RandomForestRegressor(n_jobs=-1)),
    ]
    for model in models:
        print(model[0])
        valores.append(
            (model[0], *run_reg_linear(train_x, test_x, train_y, test_y, model[1]))
        )
    valores = pd.DataFrame(valores, columns=["Model", "R2", "MSE"])
    return valores.style.background_gradient(cmap="Reds", low=0, high=1)


# In[48]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer


# In[49]:


enc = OneHotEncoder()
pca = PCA(n_components=200)
vectorizer = TfidfVectorizer(
    max_features=50000,
    min_df=10,
    ngram_range=(1, 3),
    analyzer="word",
    stop_words="english",
)


X_ohe = pca.fit_transform(
    np.concatenate(
        (
            vectorizer.fit_transform(train["name"]).toarray(),
            vectorizer.fit_transform(train["item_description"]).toarray(),
            vectorizer.fit_transform(train["name"]).toarray(),
            enc.fit_transform(
                train[
                    [
                        "brand_name",
                        # "category_name_0",
                        "category_name_1",
                        "category_name_2",
                        "shipping",
                        "item_condition_id",
                    ]
                ].values
            ).toarray(),
        ),
        axis=1,
    )
)
Y_ohe = train.log_price

print("X shape ->", X_ohe.shape)


# In[50]:


get_ipython().run_cell_magic('time', '', 'train_x_ohe, test_x_ohe, train_y_ohe, test_y_ohe = split(X_ohe, Y_ohe, True)\n\ntrain_test_show(train_x_ohe, test_x_ohe, train_y_ohe, test_y_ohe)')


# In[51]:


enc = LabelEncoder()
train_values = train[
    [
        "name",
        "brand_name",
        "category_name_0",
        "category_name_1",
        "category_name_2",
        "shipping",
        "item_condition_id",
        "item_description",
    ]
]


for col in train_values.columns:
    train_values[col] = enc.fit_transform(train_values[col])

X_le = train_values.values
Y_le = train.log_price.values
print("X shape ->", X_le.shape)


# In[52]:


get_ipython().run_cell_magic('time', '', '\ntrain_x_le, test_x_le, train_y_le, test_y_le = split(X_le, Y_le, True)\n\ntrain_test_show(train_x_le, test_x_le, train_y_le, test_y_le)')


# In[53]:


enc = LabelEncoder()
scaler = StandardScaler()

train_values = train[
    [
        "name",
        "brand_name",
        "category_name_0",
        "category_name_1",
        "category_name_2",
        "shipping",
        "item_condition_id",
        "item_description",
    ]
]


for col in train_values.columns:
    train_values[col] = scaler.fit_transform(
        enc.fit_transform(train_values[col]).reshape(-1, 1)
    )

X_le_sc = train_values.values
Y_le_sc = train.log_price.values
print("X shape ->", X_le_sc.shape)


# In[54]:


get_ipython().run_cell_magic('time', '', '\ntrain_x_le_sc, test_x_le_sc, train_y_le_sc, test_y_le_sc = split(X_le_sc, Y_le_sc, True)\n\ntrain_test_show(train_x_le_sc, test_x_le_sc, train_y_le_sc, test_y_le_sc)')


# In[55]:


from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=10, stop=100, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=50,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1,
)
# Fit the random search model
rf_random.fit(
   X_ohe, Y_ohe
)


# In[56]:


rf_random.best_params_


# In[57]:


# Plot-outputs
model = RandomForestRegressor(
    n_estimators=80,
    min_samples_split=2,
    min_samples_leaf=4,
    max_features="sqrt",
    max_depth=10,
    bootstrap=True,
)

model.fit(train_x_ohe, train_y_ohe)
y_predict = model.predict(test_x_ohe)

print("MSE: ", mean_squared_error(test_y_ohe, y_predict))
print("R2: ", r2_score(test_y_ohe, y_predict))

