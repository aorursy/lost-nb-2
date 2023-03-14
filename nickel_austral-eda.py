#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
plt.rcParams['figure.figsize'] = (30, 15)
plt.rcParams['font.size'] = 25


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


print(os.listdir("../input/avito-demand-prediction"))
print(os.listdir("../input/keras-pretrained-models"))


# In[ ]:


train = pd.read_csv("../input/avito-demand-prediction/train.csv", parse_dates=["activation_date"])
test = pd.read_csv("../input/avito-demand-prediction/test.csv", parse_dates=["activation_date"])
data = pd.concat([train, test])
del train, test


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[ ]:


data.dtypes


# In[ ]:


data.dtypes.value_counts()


# In[ ]:


data.head(5)


# In[ ]:


parent_category_name_map = {"Личные вещи" : "Personal belongings",
                            "Для дома и дачи" : "For the home and garden",
                            "Бытовая электроника" : "Consumer electronics",
                            "Недвижимость" : "Real estate",
                            "Хобби и отдых" : "Hobbies & leisure",
                            "Транспорт" : "Transport",
                            "Услуги" : "Services",
                            "Животные" : "Animals",
                            "Для бизнеса" : "For business"}

region_map = {"Свердловская область" : "Sverdlovsk oblast",
            "Самарская область" : "Samara oblast",
            "Ростовская область" : "Rostov oblast",
            "Татарстан" : "Tatarstan",
            "Волгоградская область" : "Volgograd oblast",
            "Нижегородская область" : "Nizhny Novgorod oblast",
            "Пермский край" : "Perm Krai",
            "Оренбургская область" : "Orenburg oblast",
            "Ханты-Мансийский АО" : "Khanty-Mansi Autonomous Okrug",
            "Тюменская область" : "Tyumen oblast",
            "Башкортостан" : "Bashkortostan",
            "Краснодарский край" : "Krasnodar Krai",
            "Новосибирская область" : "Novosibirsk oblast",
            "Омская область" : "Omsk oblast",
            "Белгородская область" : "Belgorod oblast",
            "Челябинская область" : "Chelyabinsk oblast",
            "Воронежская область" : "Voronezh oblast",
            "Кемеровская область" : "Kemerovo oblast",
            "Саратовская область" : "Saratov oblast",
            "Владимирская область" : "Vladimir oblast",
            "Калининградская область" : "Kaliningrad oblast",
            "Красноярский край" : "Krasnoyarsk Krai",
            "Ярославская область" : "Yaroslavl oblast",
            "Удмуртия" : "Udmurtia",
            "Алтайский край" : "Altai Krai",
            "Иркутская область" : "Irkutsk oblast",
            "Ставропольский край" : "Stavropol Krai",
            "Тульская область" : "Tula oblast"}


category_map = {"Одежда, обувь, аксессуары":"Clothing, shoes, accessories",
"Детская одежда и обувь":"Children's clothing and shoes",
"Товары для детей и игрушки":"Children's products and toys",
"Квартиры":"Apartments",
"Телефоны":"Phones",
"Мебель и интерьер":"Furniture and interior",
"Предложение услуг":"Offer services",
"Автомобили":"Cars",
"Ремонт и строительство":"Repair and construction",
"Бытовая техника":"Appliances",
"Товары для компьютера":"Products for computer",
"Дома, дачи, коттеджи":"Houses, villas, cottages",
"Красота и здоровье":"Health and beauty",
"Аудио и видео":"Audio and video",
"Спорт и отдых":"Sports and recreation",
"Коллекционирование":"Collecting",
"Оборудование для бизнеса":"Equipment for business",
"Земельные участки":"Land",
"Часы и украшения":"Watches and jewelry",
"Книги и журналы":"Books and magazines",
"Собаки":"Dogs",
"Игры, приставки и программы":"Games, consoles and software",
"Другие животные":"Other animals",
"Велосипеды":"Bikes",
"Ноутбуки":"Laptops",
"Кошки":"Cats",
"Грузовики и спецтехника":"Trucks and buses",
"Посуда и товары для кухни":"Tableware and goods for kitchen",
"Растения":"Plants",
"Планшеты и электронные книги":"Tablets and e-books",
"Товары для животных":"Pet products",
"Комнаты":"Room",
"Фототехника":"Photo",
"Коммерческая недвижимость":"Commercial property",
"Гаражи и машиноместа":"Garages and Parking spaces",
"Музыкальные инструменты":"Musical instruments",
"Оргтехника и расходники":"Office equipment and consumables",
"Птицы":"Birds",
"Продукты питания":"Food",
"Мотоциклы и мототехника":"Motorcycles and bikes",
"Настольные компьютеры":"Desktop computers",
"Аквариум":"Aquarium",
"Охота и рыбалка":"Hunting and fishing",
"Билеты и путешествия":"Tickets and travel",
"Водный транспорт":"Water transport",
"Готовый бизнес":"Ready business",
"Недвижимость за рубежом":"Property abroad"}


# In[ ]:


data['region_en'] = data['region'].apply(lambda x : region_map[x])
data['parent_category_name_en'] = data['parent_category_name'].apply(lambda x : parent_category_name_map[x])
data['category_name_en'] = data['category_name'].apply(lambda x : category_map[x])


# In[ ]:


data.head(5).T


# In[ ]:


data.isnull().sum() / data.shape[0] * 100


# In[ ]:


sns.distplot(data["deal_probability"].dropna().values, bins=120)


# In[ ]:


pd.cut(data["deal_probability"].dropna(), 10).value_counts().sort_index()


# In[ ]:


pd.cut(data["deal_probability"].dropna(), 10).value_counts(True).sort_index()


# In[ ]:


idx = range(data.deal_probability.notnull().sum())
plt.scatter(idx, np.sort(data['deal_probability'].dropna().values))


# In[ ]:


pd.concat([data.region_en.value_counts().rename("abs"), data.region_en.value_counts(True).rename("rel")], axis=1)


# In[ ]:


train = data[data.deal_probability.notnull()]


# In[ ]:


reg_label = pd.crosstab(train.region_en, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
reg_label


# In[ ]:


sns.heatmap(reg_label)


# In[ ]:


sns.boxplot(x="region_en", y="deal_probability", data=train)


# In[ ]:


cat_label = pd.crosstab(train.category_name_en, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
cat_label


# In[ ]:


train.category_name_en.value_counts()


# In[ ]:


sns.heatmap(cat_label)


# In[ ]:


sns.boxplot(x="category_name_en", y="deal_probability", data=train)


# In[ ]:


cat_label = pd.crosstab(train.parent_category_name_en, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
cat_label


# In[ ]:


data.groupby("parent_category_name_en").category_name_en.apply(set)


# In[ ]:


sns.heatmap(cat_label)


# In[ ]:


data.parent_category_name_en.value_counts(True)


# In[ ]:


sns.boxplot(x="parent_category_name_en", y="deal_probability", data=train)


# In[ ]:


data.dtypes


# In[ ]:


data.describe()


# In[ ]:


data[data.deal_probability.notnull()].activation_date.min(), data[data.deal_probability.notnull()].activation_date.max()


# In[ ]:


data[data.deal_probability.isnull()].activation_date.min(), data[data.deal_probability.isnull()].activation_date.max()


# In[ ]:


data.apply(lambda x: x.unique().shape[0])


# In[ ]:


itemsq_label = pd.crosstab(pd.qcut(train.item_seq_number, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
itemsq_label


# In[ ]:


sns.heatmap(itemsq_label)


# In[ ]:


train.groupby(pd.qcut(train.item_seq_number, 15)).deal_probability.mean()


# In[ ]:


train.groupby(pd.qcut(train.item_seq_number, 10)).deal_probability.mean().plot()


# In[ ]:


plt.scatter("item_seq_number", "deal_probability", data=train)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.crosstab')


# In[ ]:


cross = pd.crosstab(pd.qcut(train.item_seq_number, 10), train.parent_category_name_en,
            values=train.deal_probability, aggfunc=np.mean)
cross


# In[ ]:


sns.heatmap(cross)


# In[ ]:


price_label = pd.crosstab(pd.qcut(train.price, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
price_label


# In[ ]:


sns.heatmap(price_label)


# In[ ]:


user_label = pd.crosstab(train.user_type, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
user_label


# In[ ]:


sns.heatmap(user_label)


# In[ ]:


train.user_type.value_counts()


# In[ ]:


# variables de fecha
data['weekday'] = data.activation_date.dt.weekday
data['month'] = data.activation_date.dt.month
data['day'] = data.activation_date.dt.day
data['week'] = data.activation_date.dt.week 
train = data[data.deal_probability.notnull()]


# In[ ]:


temp = pd.crosstab(train.weekday, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
temp


# In[ ]:


sns.heatmap(temp)


# In[ ]:


sns.boxplot(x="weekday", y="deal_probability", data=train)


# In[ ]:


temp = pd.crosstab(train.day, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
temp


# In[ ]:


train.day.value_counts().sort_index()


# In[ ]:


train.month.value_counts()


# In[ ]:


data['description'] = data['description'].fillna("")
data['description_len'] = data['description'].apply(lambda x : len(x.split()))
train = data[data.deal_probability.notnull()]


# In[ ]:


temp = pd.crosstab(pd.qcut(train.description_len, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
temp


# In[ ]:


sns.heatmap(temp)


# In[ ]:


data['title'] = data['title'].fillna(" ")
data['title_len'] = data['title'].apply(lambda x : len(x.split()))
train = data[data.deal_probability.notnull()]


# In[ ]:


train.title_len.value_counts()


# In[ ]:


temp = pd.crosstab(train.title_len, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
temp


# In[ ]:


sns.heatmap(temp)


# In[ ]:


data['title_len_char'] = data['title'].str.len()
train = data[data.deal_probability.notnull()]


# In[ ]:


train.title_len_char.value_counts()


# In[ ]:


temp = pd.crosstab(train.title_len_char, pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
temp


# In[ ]:


sns.heatmap(temp)


# In[ ]:


train.groupby("title_len_char").deal_probability.mean()


# In[ ]:


_.plot()


# In[ ]:


train.groupby("title_len").deal_probability.mean().plot()


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


text_cols = ['description', 'param_1', 'param_2', 'param_3','title']
data["total_text"] = data[text_cols].fillna("").sum(axis=1)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

tfidf = TfidfVectorizer(ngram_range=(1,1)).fit_transform(data.total_text)


# In[ ]:


tfidf


# In[ ]:


data.index = range(data.shape[0])


# In[ ]:


from sklearn.decomposition import TruncatedSVD

n_comps = 10
tsvd = pd.DataFrame(TruncatedSVD(n_components=n_comps, algorithm='arpack').fit_transform(tfidf),
                    columns=["svd_comp_" + str(i) for i in range(n_comps)], index=data.index)
tsvd.head()


# In[ ]:


data = data.join(tsvd)
train = data[data.deal_probability.notnull()]


# In[ ]:


temp = pd.crosstab(pd.qcut(train.svd_comp_0, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
sns.heatmap(temp)
temp


# In[ ]:


temp = pd.crosstab(pd.qcut(train.svd_comp_1, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
sns.heatmap(temp)
temp


# In[ ]:


temp = pd.crosstab(pd.qcut(train.svd_comp_2, 10), pd.cut(train.deal_probability, 10)).apply(lambda x: x/x.sum(), axis=1)
sns.heatmap(temp)
temp


# In[ ]:


[train.groupby(pd.qcut(train["svd_comp_" + str(i)], 10)).deal_probability.mean().plot() for i in range(n_comps)]


# In[ ]:


print(os.listdir("../input/keras-pretrained-models"))


# In[ ]:


from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception 
from keras.applications.inception_v3 import InceptionV3
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


model = VGG16(weights=None, include_top=False)
model.summary()


# In[ ]:


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


model = ResNet50(weights=None, include_top=False)
model.summary()


# In[ ]:


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


model = Xception(weights=None, include_top=False)
model.summary()


# In[ ]:


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


model = InceptionV3(weights=None, include_top=False)
model.summary()


# In[ ]:


SVG(model_to_dot(model).create(prog='dot', format='svg'))


# In[ ]:


model = VGG16(weights=None, include_top=False)
model.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[ ]:


model.input_shape, model.output_shape


# In[ ]:


from keras.applications.vgg16 import preprocess_input
import zipfile
import cv2


# In[ ]:


myzip = zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip')
files_in_zip = myzip.namelist()
print(files_in_zip[:5])
print("total", len(files_in_zip))


# In[ ]:


myzip.close()


# In[ ]:


from time import time


# In[ ]:


# bach_size  = 100
# im_dim = 63
# names = []
# embeddings = []
# with zipfile.ZipFile('../input/avito-demand-prediction/train_jpg.zip') as f:
#     bach = np.zeros((bach_size, im_dim, im_dim, 3))
#     j = 0
#     start = time()
#     for i, name in enumerate(f.namelist()):
#         if i % 1000 == 0:
#             print("done", i, "in", time() - start)
#         if not name.endswith('.jpg'): continue
#         names.append(name.split("/")[-1].split(".")[0])
#         img = cv2.imdecode(np.frombuffer(f.read(name), dtype='uint8'), cv2.IMREAD_COLOR)
#         height, width, _ = img.shape
#         if height > width:
#             new_dim = (width*im_dim//height, im_dim)
#         else:
#             new_dim = (im_dim, height*im_dim//width)
#         img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
#         h, w = img.shape[:2]

#         off_x = (im_dim-w)//2
#         off_y = (im_dim-h)//2
#         bach[j, off_y:off_y+h, off_x:off_x+w] = img
#         j += 1
#         if j == bach_size:
#             j = 0
#             embedding = model.predict(preprocess_input(bach))
#             embeddings.append(embedding.reshape(bach_size, embedding.shape[-1]))
#             bach = np.zeros((bach_size, im_dim, im_dim,3))


# In[ ]:


# with zipfile.ZipFile('../input/avito-demand-prediction/test_jpg.zip') as f:
#     bach = np.zeros((bach_size, im_dim, im_dim, 3))
#     j = 0
#     start = time()
#     for i, name in enumerate(f.namelist()):
#         if i % 1000 == 0:
#             print("done", i, "in", time() - start)
#         if not name.endswith('.jpg'): continue
#         names.append(name.split("/")[-1].split(".")[0])
#         img = cv2.imdecode(np.frombuffer(f.read(name), dtype='uint8'), cv2.IMREAD_COLOR)
#         height, width, _ = img.shape
#         if height > width:
#             new_dim = (width*im_dim//height, im_dim)
#         else:
#             new_dim = (im_dim, height*im_dim//width)
#         img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
#         h, w = img.shape[:2]

#         off_x = (im_dim-w)//2
#         off_y = (im_dim-h)//2
#         bach[j, off_y:off_y+h, off_x:off_x+w] = img
#         j += 1
#         if j == bach_size:
#             j = 0
#             embedding = model.predict(preprocess_input(bach))
#             embeddings.append(embedding.reshape(bach_size, embedding.shape[-1]))
#             bach = np.zeros((bach_size, im_dim, im_dim,3))


# In[ ]:


# embeddings = np.concatenate(embeddings)
# embeddings = pd.DataFrame(embeddings, index=names[:embeddings.shape[0]], columns=["img_emb_" + str(i) for i in range(embeddings.shape[1])])


# In[ ]:


# embeddings


# In[ ]:


# data = data.join(embeddings, on="image")


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data = data.drop(['activation_date', 'image', 'title', 'description', 'region_en',
                  'parent_category_name_en', 'category_name_en', 'total_text'], axis=1)
data.info()


# In[ ]:


data.set_index("item_id", inplace=True)
obj_columns = data.select_dtypes("object").columns
obj_columns


# In[ ]:


for c in obj_columns:
    data[c] = pd.factorize(data[c])[0]


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum(axis=0)


# In[ ]:


test = data[data.deal_probability.isnull()].drop("deal_probability", axis=1)
train = data.drop(test.index)
del data


# In[ ]:


from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor


# In[ ]:


get_ipython().run_line_magic('pinfo', 'LGBMRegressor.fit')


# In[ ]:


def train_CV(train, test, folds, **params):
    test_preds = []
    train_preds = []
    for i, (train_idx, test_idx) in enumerate(folds):
        X_train = train.drop("deal_probability", axis=1).iloc[train_idx]
        y_train = train.deal_probability.iloc[train_idx]
        X_valid = train.drop("deal_probability", axis=1).iloc[test_idx]
        y_valid = train.deal_probability.iloc[test_idx]
        learner = LGBMRegressor(n_estimators=10000, **params)
        learner.fit(X_train, y_train, early_stopping_rounds=10,
                    eval_metric="rmse", verbose=100,
                    eval_set=[(X_train, y_train),
                              (X_valid, y_valid)]
                   )
        preds = pd.Series(learner.predict(X_valid), index=X_valid.index, name="deal_probability")
        train_preds.append(preds)
        preds = pd.Series(learner.predict(test), index=test.index, name="fold_" + str(i))
        test_preds.append(preds)
    return pd.concat(train_preds).clip(0, 1),           pd.concat(test_preds, axis=1).mean(axis=1).clip(0, 1).rename("deal_probability")


# In[ ]:


folds = list(KFold(n_splits=5, shuffle=True).split(train))
res = pd.Series(index=[2 ** i for i in range(5, 11)])
best_res = 10
for nl in res.index:
    print("*" * 20)
    print("doing", nl)
    train_preds, test_preds = train_CV(train, test, folds, num_leaves=nl)
    res = np.power(np.power(train.deal_probability - train_preds.loc[train.index], 2).mean(), 0.5)
    if res < best_res:
        print("*" * 20)
        print("*" * 20)
        print("got best with nl {}: {}".format(nl, res))
        print("*" * 20)
        print("*" * 20)
        best_res = res
        best_test_preds = test_preds


# In[ ]:


train.deal_probability.mean(), best_test_preds.mean()


# In[ ]:


best_test_preds.to_csv("preds.csv", header=True, index=True)

