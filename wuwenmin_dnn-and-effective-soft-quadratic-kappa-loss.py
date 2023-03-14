#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.metrics import Metric
import tensorflow.keras.backend as K
RANDOM_SEED = 3
tf.random.set_seed(RANDOM_SEED)
tf.keras.backend.set_floatx('float64')

@tf.function
def cohen_kappa_loss(y_true, y_pred, row_label_vec, col_label_vec, weight_mat,  eps=1e-6, dtype=tf.float64):
    labels = tf.matmul(y_true, col_label_vec)
    weight = tf.pow(tf.tile(labels, [1, tf.shape(y_true)[1]]) - tf.tile(row_label_vec, [tf.shape(y_true)[0], 1]), 2)
    weight /= tf.cast(tf.pow(tf.shape(y_true)[1] - 1, 2), dtype=dtype)
    numerator = tf.reduce_sum(weight * y_pred)
    
    denominator = tf.reduce_sum(
        tf.matmul(
            tf.reduce_sum(y_true, axis=0, keepdims=True),
            tf.matmul(weight_mat, tf.transpose(tf.reduce_sum(y_pred, axis=0, keepdims=True)))
        )
    )
    
    denominator /= tf.cast(tf.shape(y_true)[0], dtype=dtype)
    
    return tf.math.log(numerator / denominator + eps)

class CohenKappaLoss(tf.keras.losses.Loss):
    def __init__(self,
                 num_classes,
                 name='cohen_kappa_loss',
                 eps=1e-6,
                 dtype=tf.float64):
        super(CohenKappaLoss, self).__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        
        self.num_classes = num_classes
        self.eps = eps
        self.dtype = dtype
        label_vec = tf.range(num_classes, dtype=dtype)
        self.row_label_vec = tf.reshape(label_vec, [1, num_classes])
        self.col_label_vec = tf.reshape(label_vec, [num_classes, 1])
        self.weight_mat = tf.pow(
            tf.tile(self.col_label_vec, [1, num_classes]) - tf.tile(self.row_label_vec, [num_classes, 1]),
        2) / tf.cast(tf.pow(num_classes - 1, 2), dtype=dtype)


    def call(self, y_true, y_pred, sample_weight=None):
        return cohen_kappa_loss(
            y_true, y_pred, self.row_label_vec, self.col_label_vec, self.weight_mat, self.eps, self.dtype
        )


    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "eps": self.eps,
            "dtype": self.dtype
        }
        base_config = super(CohenKappaLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[2]:


class CohenKappa(Metric):
    """
    This metric is copied from TensorFlow Addons
    """
    def __init__(self,
                 num_classes,
                 name='cohen_kappa',
                 weightage=None,
                 dtype=tf.float32):
        super(CohenKappa, self).__init__(name=name, dtype=dtype)

        if weightage not in (None, 'linear', 'quadratic'):
            raise ValueError("Unknown kappa weighting type.")
        else:
            self.weightage = weightage

        self.num_classes = num_classes
        self.conf_mtx = self.add_weight(
            'conf_mtx',
            shape=(self.num_classes, self.num_classes),
            initializer=tf.keras.initializers.zeros,
            dtype=tf.int32)
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 2:
            y_true = tf.argmax(y_true, axis=1)
        if len(y_pred.shape) == 2:
            y_pred = tf.argmax(y_pred, axis=1)
        
        y_true = tf.cast(y_true, dtype=tf.int32)
        y_pred = tf.cast(y_pred, dtype=tf.int32)
        
        if y_true.shape.as_list() != y_pred.shape.as_list():
            raise ValueError(
                "Number of samples in y_true and y_pred are different")

        # compute the new values of the confusion matrix
        new_conf_mtx = tf.math.confusion_matrix(
            labels=y_true,
            predictions=y_pred,
            num_classes=self.num_classes,
            weights=sample_weight)

        # update the values in the original confusion matrix
        return self.conf_mtx.assign_add(new_conf_mtx)
    
    def result(self):
        nb_ratings = tf.shape(self.conf_mtx)[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.int32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.int32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        else:
            weight_mtx += tf.range(nb_ratings, dtype=tf.int32)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

            if self.weightage == 'linear':
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)
            weight_mtx = tf.cast(weight_mtx, dtype=tf.float32)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] *                     actual_ratings_hist[None, ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=tf.float32)
        out_prod = tf.cast(out_prod, dtype=tf.float32)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        kp = 1 - (numerator / denominator)
        return kp
    
    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
            "weightage": self.weightage,
        }
        base_config = super(CohenKappa, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def reset_states(self):
        """Resets all of the metric state variables."""

        for v in self.variables:
            K.set_value(
                v, np.zeros((self.num_classes, self.num_classes), np.int32))


# In[3]:


import re
import pandas as pd
import numpy as np
import gc
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Tuple

DAY_OF_WEEKS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_OF_WEEKS_MAP = {day: i for (i, day) in enumerate(DAY_OF_WEEKS)}

def gen_user_samples(user_events: pd.DataFrame, num2title: dict, assess_num_lis:list, title2win_code: dict,
                     event_code_list: list, specs: pd.DataFrame=None, is_test_set=False) -> list:
    user_samples = []
    last_type = 0
    types_count = {'Clip':0, 'Activity':0, 'Assessment':0, 'Game':0}

    # time_first_activity = float(user_events['timestamp'].values[0])
    time_spent_each_title = {f'time_spent_title{title_num}':0 for title_num in num2title}
    event_code_count = {f'event_{code}_cnt':0 for code in event_code_list}
    accuracy_group_cnt = {f'acc_grp_{grp}_cnt':0 for grp in [0, 1, 2, 3] }

    atmpts_each_assess = {f'atmpts_each_assess{assess_num}': 0 for assess_num in assess_num_lis}
    wins_each_assess = {f'wins_each_assess{assess_num}': 0 for assess_num in assess_num_lis}
    losses_each_assess = {f'losses_each_assess{assess_num}': 0 for assess_num in assess_num_lis}

    accumu_acc_grp = 0
    accumu_acc = 0
    accumu_win_n = 0
    accumu_loss_n = 0
    accumu_actions = 0
    durations = []
    non_assess_durations = []
    counter = 0

    for session_id, session in user_events.groupby('game_session', sort=False):
        # sort inside to achieve better performace
        session = session.sort_values(by='timestamp')
        session_type = session['type'].iloc[0]
        session_title = session['title'].iloc[0]

        if session_type != 'Assessment':
            time_spent = int(session['game_time'].iloc[-1] / 1000)
            time_spent_each_title[f'time_spent_title{session_title}'] += time_spent
            non_assess_durations.append((session.iloc[-1]['timestamp'] - session.iloc[0]['timestamp']).seconds)

        if (session_type == 'Assessment') & (is_test_set or len(session) > 1):
            # all_4100 = session.query(f'event_code == {title2win_code[session_title]}')
            all_4100 = session[session['event_code'] == title2win_code[session_title]]
            #numbers of wins and losses (globally)
            # TODO: count on each title since some of them maybe similar
            win_n = all_4100['event_data'].str.contains('true').sum()
            loss_n = all_4100['event_data'].str.contains('false').sum()

            # init feature then update
            features = types_count.copy()
            features['installation_id'] = session['installation_id'].iloc[-1]
            features.update(time_spent_each_title.copy())
            features.update(event_code_count.copy())
            features.update(atmpts_each_assess.copy())
            features.update(wins_each_assess.copy())
            features.update(losses_each_assess.copy())
            features['session_title'] = session_title
            features['accumu_win_n'] = accumu_win_n
            features['accumu_loss_n'] = accumu_loss_n
            accumu_win_n += win_n
            accumu_loss_n += loss_n
            atmpts_each_assess[f'atmpts_each_assess{session_title}'] += 1
            wins_each_assess[f'wins_each_assess{session_title}'] += win_n
            losses_each_assess[f'losses_each_assess{session_title}'] += loss_n

            features['day_of_the_week'] = DAY_OF_WEEKS_MAP[(session['timestamp'].iloc[-1]).strftime("%A")]
            features['hour'] = session['timestamp'].iloc[-1].hour
            features['month'] = session['timestamp'].iloc[-1].month

            if durations == []:
                features['duration_mean'] = 0
                features['duration_sum'] = 0
                features['duration_std'] = 0
                features['duration_min'] = 0
                features['duration_max'] = 0
            else:
                features['duration_mean'] = np.mean(durations)
                features['duration_sum'] = np.sum(durations)
                features['duration_std'] = np.std(durations)
                features['duration_min'] = np.min(durations)
                features['duration_max'] = np.max(durations)
            durations.append((session.iloc[-1]['timestamp'] - session.iloc[0]['timestamp']).seconds)

            if non_assess_durations == []:
                features['non_assess_duration_mean'] = 0
                features['non_assess_duration_sum'] = 0
                features['non_assess_duration_std'] = 0
                features['non_assess_duration_min'] = 0
                features['non_assess_duration_max'] = 0
            else:
                features['non_assess_duration_mean'] = np.mean(non_assess_durations)
                features['non_assess_duration_sum'] = np.sum(non_assess_durations)
                features['non_assess_duration_std'] = np.std(non_assess_durations)
                features['non_assess_duration_min'] = np.min(non_assess_durations)
                features['non_assess_duration_max'] = np.max(non_assess_durations)


            # average of the all accuracy of this player
            features['accuracy_ave'] = accumu_acc / counter if counter > 0 else 0
            accuracy = win_n / (win_n + loss_n) if (win_n + loss_n) > 0 else 0
            accumu_acc += accuracy
            if accuracy == 0:
                features['accuracy_group'] = 0
            elif accuracy == 1:
                features['accuracy_group'] = 3
            elif accuracy == 0.5:
                features['accuracy_group'] = 2
            else:
                features['accuracy_group'] = 1
            features.update(accuracy_group_cnt.copy())
            accuracy_group_cnt['acc_grp_{}_cnt'.format(features['accuracy_group'])] += 1
            # average of accuracy_groups of this player
            features['accuracy_group_ave'] = accumu_acc_grp / counter if counter > 0 else 0
            accumu_acc_grp += features['accuracy_group']


            # how many actions the player has done in this game_session
            features['accumu_actions'] = accumu_actions

            # if test_set, all sessions belong to the final dataset
            # elif train, needs to be passed throught this clausule

            if is_test_set or (win_n + loss_n) > 0:
                user_samples.append(features)

            counter += 1

        # how many actions was made in each event_code
        event_codes = Counter(session['event_code'])
        for key in event_codes.keys():
            event_code_count[f'event_{key}_cnt'] += event_codes[key]

        # how many actions the player has done
        accumu_actions += len(session)
        if last_type != session_type:
            types_count[session_type] += 1
            last_type = session_type

    # if test_set, only the last assessment must be predicted,
    # the previous are scraped
    if is_test_set:
        return user_samples[-1]
    return user_samples

def gen_assess_avg_acc(train_labels, title2num):
    """
    Generate average accuracy of each assessment
    """
    print('Calculate average accuracy of each assessment')
    return {
        title2num[title]: group['accuracy'].mean() \
        for (title, group) in train_labels.groupby('title', sort=False)
    }

def gen_assess_corr_rate(train_labels, title2num):
    """
    Generate correct rate of each assessment
    """
    print('Calculate correct rate of each assessment')
    df = train_labels
    acc_assessment_dict = dict()
    for title, group in df.groupby('title'):
        num_correct = group['num_correct'].sum()
        num_incorrect = group['num_incorrect'].sum()
        acc_assessment_dict[title2num[title]] = num_correct / (num_correct + num_incorrect + 1E-6)
    return acc_assessment_dict


def gen_data_sets(train_events: pd.DataFrame, test_events: pd.DataFrame, train_labels: pd.DataFrame,
                  spec_data: pd.DataFrame = None, n_jobs=-1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # unique title list
    print('starting generate data sets...')

    title_list = np.unique(np.hstack([
        train_events['title'].values, test_events['title'].values
    ])).tolist()

    # num <-> title
    # num <-> title
    title2num = {title: num for (num, title) in enumerate(title_list)}
    num2title = {num: title for (num, title) in enumerate(title_list)}

    assess_avg_acc = gen_assess_avg_acc(train_labels, title2num)
    assess_corr_rate = gen_assess_corr_rate(train_labels, title2num)

    assess_titles = np.unique(
        np.hstack([
            train_events[train_events['type'] == 'Assessment']['title'].values,
            test_events[test_events['type'] == 'Assessment']['title'].values,
        ])
    )
    print(f'assessment titles: {assess_titles}')
    assess_num_lis = [title2num[title] for title in assess_titles]
    print(f'assessment num list: {assess_num_lis}')

    # title num to event code
    title2win_code = {num : 4100 for num in num2title}
    title2win_code[title2num['Bird Measurer (Assessment)']] = 4110

    # unique event code list
    event_code_list = np.unique(np.hstack([
        train_events['event_code'].values, test_events['event_code'].values
    ])).tolist()

    # map title to title number
    train_events['title'] = train_events['title'].map(dict(title2num)).astype(np.int16)
    test_events['title'] = test_events['title'].map(dict(title2num)).astype(np.int16)

    train_events['timestamp'] = pd.to_datetime(train_events['timestamp'])
    test_events['timestamp'] = pd.to_datetime(test_events['timestamp'])

    print('start generating samples...')
    num_process = cpu_count() if n_jobs == -1 else n_jobs
    with Pool(processes=num_process) as pool:
        processed_users = 0
        train_samples = []
        res = [pool.apply_async(gen_user_samples, args=(events, num2title, assess_num_lis, title2win_code, event_code_list))                for (install_id, events) in train_events.groupby('installation_id')]
        for rr in res:
            train_samples += rr.get()
            processed_users += 1
            if not (processed_users % 1000):
                print(f'Proessed {processed_users} users in train')

        test_samples = []
        processed_users = 0
        res = [pool.apply_async(gen_user_samples, args=(events, num2title, assess_num_lis, title2win_code, event_code_list, None, True))                for (install_id, events) in test_events.groupby('installation_id')]
        for rr in res:
            test_samples.append(rr.get())
            processed_users += 1
            if not (processed_users % 100):
                print(f'Proessed {processed_users} users in test')

        train_output, test_output = pd.DataFrame(train_samples), pd.DataFrame(test_samples)
        train_output['assess_avg_acc'] = train_output['session_title']                                          .map(assess_avg_acc).astype(np.float32)
        train_output['assess_corr_rate'] = train_output['session_title']                                          .map(assess_corr_rate).astype(np.float32)

        test_output['assess_avg_acc'] = test_output['session_title']                                          .map(assess_avg_acc).astype(np.float32)
        test_output['assess_corr_rate'] = test_output['session_title']                                          .map(assess_corr_rate).astype(np.float32)
    return train_output, test_output


# In[4]:


data_dir = '../input/data-science-bowl-2019'

# specify column type to reduce memory usage
COL_TYPES = {
    'game_session': 'object',
    'timestamp': 'object',
    'event_data': 'object',
    'installation_id': 'object',
    'title': 'category',
    'type': 'category',
    'game_time': 'int64',
    'event_code': 'int32'
}
use_cols = list(COL_TYPES.keys())
print(f'USE COLS: {use_cols}')
print('loading train data...')
train = pd.read_csv(f'{data_dir}/train.csv', usecols=use_cols, dtype=COL_TYPES)
print('loading test data...')
test = pd.read_csv(f'{data_dir}/test.csv', usecols=use_cols, dtype=COL_TYPES)

print('loading train label data...')
train_labels = pd.read_csv(f'{data_dir}/train_labels.csv', usecols=['title', 'num_correct', 'num_incorrect', 'accuracy'],
                           dtype={'title': 'object', 'num_correct': np.int32, 'num_incorrect': np.int32, 'accuracy': np.float32})
train_set, test_set = gen_data_sets(train, test, train_labels)
del train
del test
gc.collect()


# In[5]:


category_features = ['session_title','day_of_the_week', 'hour', 'month']
feature_blacklist = frozenset(['accuracy_group', 'installation_id', 'game_session'])
all_features = [col for col in train_set.columns if col not in  feature_blacklist]
multi_val_fea = [col for col in all_features if train_set[col].nunique() > 1]
print(f'totally {len(all_features)}, {len(multi_val_fea)} of them have multiple values')

used_features = [fea for fea in multi_val_fea if fea not in feature_blacklist]
print(f'totally {len(used_features)} features are used for training')
need_log_pat = re.compile(r'time_spent.*|duration.*|non_assess_duration.*')
need_log_fea = [fea for fea in used_features if need_log_pat.match(fea)]
print(f'{len(need_log_fea)} need log features: \n{need_log_fea}')
numeric_fea = [fea for fea in used_features if fea not in category_features]
print(f'{len(numeric_fea)} numeric features: \n{numeric_fea}')

tmp_df  = pd.concat([train_set[used_features], test_set[used_features]])
feature2vocab = {}
for feature in category_features:
    feature2vocab[feature] = np.unique(tmp_df[feature].values).tolist()
train_df = tmp_df.iloc[:len(train_set), :].copy()
train_df['label'] = train_set['accuracy_group']
test_df = tmp_df.iloc[len(train_set):, :].copy()
submission = test_set[['installation_id']].copy()


# In[6]:


def df_to_dataset(dataframe, shuffle=True, batch_size=32, num_classes=None):
    dataframe = dataframe.copy()
    if 'label' in dataframe.columns:
        labels = dataframe.pop('label')
        if num_classes and num_classes > 2:
            labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(dict(dataframe))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds

def build_nn_model(feature_columns, num_classes=None, dtype=tf.float64):
    dtype = tf.float64
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns, dtype=dtype)
    model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(1024, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(512, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(256, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(128, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(64, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(32, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
        tf.keras.layers.Dense(16, activation='relu', dtype=dtype),
        tf.keras.layers.BatchNormalization(dtype=dtype),
    ])
    if not num_classes:
        model.add(tf.keras.layers.Dense(1, dtype=dtype))
    elif num_classes == 2:
        model.add(tf.keras.layers.Dense(1, activation='sigmoid',dtype=dtype))
    else:
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax', dtype=dtype))
    return model


# In[7]:


from sklearn.preprocessing import MinMaxScaler

train_df[need_log_fea] = np.log(train_df[need_log_fea].values + 1.0)
test_df[need_log_fea] = np.log(test_df[need_log_fea].values + 1.0)
scaler = MinMaxScaler()
train_df[numeric_fea] = scaler.fit_transform(train_df[numeric_fea].values.astype(np.float64))
test_df[numeric_fea] = scaler.transform(test_df[numeric_fea].values.astype(np.float64))

emb_size = 32
feature_columns = []
for fea in used_features:
    if fea in category_features:
        categorical_col = tf.feature_column.categorical_column_with_vocabulary_list(fea, feature2vocab[fea])
        feature_columns.append(tf.feature_column.embedding_column(categorical_col, emb_size))
    else:
        feature_columns.append(tf.feature_column.numeric_column(fea))


# In[8]:


from sklearn.utils import shuffle

install_ids = train_set['installation_id'].copy()
train_df, install_ids = shuffle(train_df, install_ids, random_state=RANDOM_SEED)
train_df.reset_index(inplace=True, drop=True)

num_epoch = 64
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
val_kappas = []
test_set_ = df_to_dataset(test_df, shuffle=False, num_classes=4, batch_size=test_df.shape[0])

test_preds = np.zeros((test_df.shape[0], 4))
for train_idx, val_idx in gkf.split(train_df, groups=install_ids):
    train_set_ = df_to_dataset(train_df.iloc[train_idx].copy(), num_classes=4, batch_size=64)
    val_set_ = df_to_dataset(train_df.iloc[val_idx].copy(), shuffle=False, num_classes=4, batch_size=64)
    model = build_nn_model(feature_columns, num_classes=4)
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(),
        loss=CohenKappaLoss(4),
        metrics=[CohenKappa(num_classes=4, weightage='quadratic')]
    )
    model.fit(train_set_, epochs=num_epoch, verbose=2)
    loss, kappa = model.evaluate(val_set_, verbose=2)
    val_kappas.append(kappa)
    print(f'validation result, loss: {loss}, kappa: {kappa}')
    test_preds += model.predict(test_set_)
print(f'validation mean: {np.mean(val_kappas)}, std: {np.std(val_kappas)}')


# In[9]:


preds = test_preds.argmax(axis=1).astype(np.int8)
print(f'predicted accuracy_group distribution:\n\n{pd.Series(preds).value_counts(normalize=True)} \n\n')
submission['accuracy_group'] = preds
submission.to_csv('submission.csv', index=False)

