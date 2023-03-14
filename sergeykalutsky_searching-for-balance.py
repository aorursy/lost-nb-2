#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set()
sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[2]:


train = pd.read_csv('../input/train.csv')


# In[3]:


# Those are key identities model's bias of which we need to minimize
key_ident = ['male',
             'female',
             'homosexual_gay_or_lesbian',
             'christian',
             'jewish',
             'muslim',
             'black',
             'white',
             'psychiatric_or_mental_illness']


# In[4]:


# Subset of data labeled based on mentioned identity of the text
identity_df = train.iloc[:, train.columns != 'parent_id'].dropna()
print('labeled identity samples:', identity_df.shape[0])


# In[5]:


identity_df = train[key_ident + ['comment_text', 'target']].dropna()
for identity in key_ident:
    identity_df[identity] = identity_df[identity]                            .apply(lambda x: 1 if x >= 0.5 else 0)
identity_df['sum'] = identity_df[key_ident].sum(axis=1)
identity_df['sum'].value_counts().plot(kind='bar').set_title('number of key identities per comment')


# In[6]:


# Somehow this comment managed to mention almost all of them
identity_df[identity_df['sum'] == 8]['comment_text'].values[0].replace('\n', ' ')


# In[7]:


groups_sampels = identity_df[key_ident].sum()
y = groups_sampels.index.values
x = groups_sampels.values
sns.barplot(x=x, y=y).set_title('# of key identity samples')


# In[8]:


identity_df['target'] = identity_df['target'].apply(lambda x: 1 if x >= 0.5 else 0)
insults = []
for ident in key_ident:
    total = identity_df[(identity_df[ident] == 1) & (identity_df['target'] == 1)].shape[0]
    insults.append(total)
df = pd.DataFrame(data={'toxic':np.array(insults), 'total': x, 'group':y})
sns.barplot(x='toxic', y='group', data=df).set_title('# of hatefull comments per group')


# In[9]:


sns.barplot(x='total', y='group', data=df, color = "red")
sns.barplot(x='toxic', y='group', data=df, color = "#0000A3").set_title('toxic to all comments')


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TweetTokenizer
from sklearn import metrics


# In[11]:


# Define bias metrics, then evaluate our new model for bias using the validation set predictions
# https://www.kaggle.com/dborkan/benchmark-kernel
SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples 
        and the background positive examples.
    """
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples 
        and the background negative examples.
    """
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups 
      and one model.
    """
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, 
                                                    subgroup, 
                                                    label_col, 
                                                    model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, 
                                            subgroup, 
                                            label_col, 
                                            model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, 
                                            subgroup, 
                                            label_col, 
                                            model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)
#-----------------------------------------------------------------------------------------------------
# Calculate the final score

def calculate_overall_auc(df, oof_name):
    true_labels = df['target']
    predicted_labels = df[oof_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


# In[12]:


# adding preprocessing from this kernel: https://www.kaggle.com/taindow/simple-cudnngru-python-keras
punct_mapping = {"_":" ", "`":" "}
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
def clean_special_chars(text, punct, mapping):
    for p in mapping:
        text = text.replace(p, mapping[p])    
    for p in punct:
        text = text.replace(p, f' {p} ')     
    return text
identity_df['comment_text'] = identity_df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
identity_df['comment_text'] = identity_df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))


# In[13]:


# Our Model and predict function
# Taken from https://www.kaggle.com/artgor/toxicity-eda-model-interpretation-and-more
logreg = LogisticRegression()
oof_name = 'predicted_target'

def fit_log_reg(X_train, y_train, valid_df):
    logreg.fit(X_train, y_train)
    valid_df[oof_name] = logreg.predict_proba(valid_vectorized)[:, 1]
    return valid_df


# In[14]:


train_df, valid_df = train_test_split(identity_df, test_size=0.2)
y_valid = valid_df['target']
for col in key_ident + ['target']:
    valid_df[col] = np.where(valid_df[col] >= 0.5, True, False)


# In[15]:


get_ipython().run_cell_magic('time', '', "tokenizer = TweetTokenizer()\nvectorizer = TfidfVectorizer(ngram_range=(1, 2), \n                             tokenizer=tokenizer.tokenize, \n                             max_features=30000)\nvectorizer.fit(identity_df['comment_text'].values)\nvalid_vectorized = vectorizer.transform(valid_df['comment_text'].values)")


# In[16]:


# Select only examples with toxic comments mentioning identity
women_negative = train_df[~((train_df['female'] == True) & 
                            (train_df['target'] == False))]
y_train = women_negative['target']
train_vectorized = vectorizer.transform(women_negative['comment_text'].values)


# In[17]:


track_column = ['female']
valid_df = fit_log_reg(train_vectorized, y_train, valid_df)
pretty_cols = ['subgroup', 'subgroup_auc', 'bpsn_auc', 'bnsp_auc', 'subgroup_size']

def get_scores(valid_df, track_column, oof_name):
    bias_metrics_df = compute_bias_metrics_for_model(valid_df, 
                                                     track_column, 
                                                     oof_name, 'target')
    final_metric = get_final_metric(bias_metrics_df, 
                                    calculate_overall_auc(valid_df, oof_name))
    
    return bias_metrics_df[pretty_cols], final_metric

metrics_df, final_metric = get_scores(valid_df, track_column, oof_name)
roc_auc = metrics.roc_auc_score(valid_df[track_column], valid_df['predicted_target'])
print('bias score:', final_metric)
print('classic roc_auc_score:', roc_auc)
metrics_df


# In[18]:


# Now select only examples without toxic coments mentioning the identity
women_pos = train_df[((train_df['female'] == True) & (train_df['target'] == False))]
# We will inject this data in 10 steps
n = 10
split = women_pos.shape[0]//n
dfs = [women_pos.iloc[i*split:(i+1)*split].copy() for i in range(n)]


# In[19]:


# Injecting positive comments, mentioning identity and retraining model each time
metrics_dfs = [metrics_df]
final_metrics = [(final_metric, roc_auc)]
for i in range(n):
    women_negative = pd.concat([women_negative, dfs[i]])
    train_vectorized = vectorizer.transform(women_negative['comment_text'].values)
    y_train = women_negative['target']
    valid_df = fit_log_reg(train_vectorized, y_train, valid_df)
    metrics_df, final_metric = get_scores(valid_df, track_column, oof_name)
    roc_auc = metrics.roc_auc_score(valid_df[track_column], valid_df['predicted_target'])
    metrics_dfs.append(metrics_df)
    final_metrics.append((final_metric, roc_auc))
    print(final_metric, roc_auc)
    print(metrics_df)


# In[20]:


graf = pd.concat(metrics_dfs)
graf['data injection steps'] = np.arange(len(metrics_dfs))
graf.set_index(['data injection steps'], inplace = True)
graf[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].plot()


# In[21]:


plt.plot(*zip(*final_metrics))
plt.xlabel('Competition metrics')
plt.ylabel('Standard roc_auc')


# In[22]:


# Select only examples with toxic comments related to key identities
related = train_df[~(train_df['sum'] == 0)]
y_train = related['target']
train_vectorized = vectorizer.transform(related['comment_text'].values)


# In[23]:


valid_df = fit_log_reg(train_vectorized, y_train, valid_df)
metrics_df, final_metric = get_scores(valid_df, track_column, oof_name)
roc_auc = metrics.roc_auc_score(valid_df[track_column], valid_df['predicted_target'])
print('bias score:', final_metric)
print('classic roc_auc_score:', roc_auc)
metrics_df


# In[24]:


# Now select only examples of unrelated to key identity comments
unrelated = train_df[(train_df['sum'] == 0)]
# We will inject this data in 10 steps
n = 10
split = unrelated.shape[0]//n
dfs = [unrelated.iloc[i*split:(i+1)*split].copy() for i in range(n)]


# In[25]:


# Injecting unrelated comments
metrics_dfs = [metrics_df]
final_metrics = [(final_metric, roc_auc)]
for i in range(n):
    related = pd.concat([related, dfs[i]])
    train_vectorized = vectorizer.transform(related['comment_text'].values)
    y_train = related['target']
    valid_df = fit_log_reg(train_vectorized, y_train, valid_df)
    metrics_df, final_metric = get_scores(valid_df, track_column, oof_name)
    roc_auc = metrics.roc_auc_score(valid_df[track_column], valid_df['predicted_target'])
    metrics_dfs.append(metrics_df)
    final_metrics.append((final_metric, roc_auc))
    print(final_metric, roc_auc)
    print(metrics_df)


# In[26]:


graf = pd.concat(metrics_dfs)
graf['data injection steps'] = np.arange(len(metrics_dfs))
graf.set_index(['data injection steps'], inplace = True)
graf[['subgroup_auc', 'bpsn_auc', 'bnsp_auc']].plot()


# In[27]:


plt.plot(*zip(*final_metrics))
plt.xlabel('Competition metrics')
plt.ylabel('Standard roc_auc')

