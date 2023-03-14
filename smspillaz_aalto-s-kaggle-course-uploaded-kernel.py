#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display
import datetime
import gc
import itertools
import json
import operator
import os
import pandas as pd
import pickle
import pprint
import numpy as np
import re
import seaborn as sns
import spacy
import torch
import torch.optim as optim

from collections import Counter, deque
from pytorch_pretrained_bert import BertAdam
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    make_scorer,
    mean_squared_error
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from skorch.callbacks import (
    Callback,
    Checkpoint,
    EpochScoring,
    LRScheduler,
    ProgressBar,
    TrainEndCheckpoint
)

from utils.callbacks import SummarizeParameters
from utils.data import load_training_test_data
from utils.dataframe import (
    categories_from_column,
    column_list_to_category_flags,
    count_json_in_dataframes,
    count_ngrams_up_to_n,
    drop_columns_from_dataframes,
    map_categorical_column_to_category_ids,
    normalize_categories,
    normalize_description,
    numerical_feature_engineering_on_dataframe,
    parse_address_components,
    remove_outliers,
    remap_column,
    remap_columns_with_transform,
    remap_date_column_to_days_before,
    remove_small_or_stopwords_from_ranking
)
from utils.featurize import (
    featurize_for_tabular_models,
    featurize_for_tree_models,
)
from utils.gc import gc_and_clear_caches
from utils.doc2vec import (
    column_to_doc_vectors
)
from utils.model import (
    basic_logistic_regression_pipeline,
    basic_xgboost_pipeline,
    basic_adaboost_pipeline,
    basic_extratrees_pipeline,
    basic_svc_pipeline,
    basic_random_forest_pipeline,
    expand_onehot_encoding,
    format_statistics,
    get_prediction_probabilities_with_columns,
    get_prediction_probabilities_with_columns_from_predictions,
    prediction_accuracy,
    write_predictions_table_to_csv,
    rescale_features_and_split_into_continuous_and_categorical,
    split_into_continuous_and_categorical,
    test_model_with_k_fold_cross_validation,
    train_model_and_get_validation_and_test_set_predictions
)
from utils.language_models.bert import (
    BertClassifier,
    BertForSequenceClassification,
    TensorTuple,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_PROPORTION,
    bert_featurize_data_frames,
    create_bert_model,
    create_bert_model_with_tabular_features,
)
from utils.language_models.descriptions import (
    descriptions_to_word_sequences,
    generate_bigrams,
    generate_description_sequences,
    maybe_cuda,
    postprocess_sequences,
    token_dictionary_seq_encoder,
    tokenize_sequences,
    torchtext_create_text_vocab,
    torchtext_process_texts,
    words_to_one_hot_lookups,
)
from utils.language_models.featurize import (
    featurize_sequences_from_dataframe,
    featurize_sequences_from_sentence_lists,
)
from utils.language_models.fasttext import (
    FastText,
    FastTextWithTabularData
)
from utils.language_models.simple_rnn import (
    CheckpointAndKeepBest,
    LRAnnealing,
    NoToTensorInLossClassifier,
    SimpleRNNPredictor,
    SimpleRNNTabularDataPredictor
)
from utils.language_models.split import (
    shuffled_train_test_split_by_indices,
    simple_train_test_split_without_shuffle_func,
    ordered_train_test_split_with_oversampling
)
from utils.language_models.textcnn import (
    TextCNN,
    TextCNNWithTabularData
)
from utils.language_models.ulmfit import (
    load_ulmfit_classifier_with_transfer_learning_from_data_frame,
    train_ulmfit_model_and_get_validation_and_test_set_predictions,
    train_ulmfit_classifier_with_gradual_unfreezing
)
from utils.language_models.visualization import (
    preview_tokenization,
    preview_encoded_sentences
)
from utils.report import (
    generate_classification_report_from_preds,
    generate_classification_report
)

nlp = spacy.load("en")


# In[2]:


torch.cuda.is_available()


# In[3]:


(ALL_TRAIN_DATAFRAME, TEST_DATAFRAME) =   load_training_test_data(os.path.join('data', 'train.json'),
                          os.path.join('data', 'test.json'))
TRAIN_INDEX, VALIDATION_INDEX = train_test_split(ALL_TRAIN_DATAFRAME.index, test_size=0.1)
TRAIN_DATAFRAME = ALL_TRAIN_DATAFRAME.iloc[TRAIN_INDEX].reset_index()
VALIDATION_DATAFRAME = ALL_TRAIN_DATAFRAME.iloc[VALIDATION_INDEX].reset_index()
TEST_DATAFRAME = TEST_DATAFRAME.reset_index(drop=True)


# In[4]:


ALL_TRAIN_DATAFRAME.head()


# In[5]:


ALL_TRAIN_DATAFRAME.describe()


# In[6]:


TEST_DATAFRAME.head()


# In[7]:


TEST_DATAFRAME.describe()


# In[8]:


CORE_NUMERICAL_COLUMNS = ['bathrooms', 'bedrooms', 'price', 'latitude', 'longitude']


# In[9]:


NUMERICAL_QUANTILES = {
    'bathrooms': (0.0, 0.999),
    'bedrooms': (0.0, 0.999),
    'latitude': (0.01, 0.99),
    'longitude': (0.01, 0.99),
    'price': (0.01, 0.99)
}


# In[10]:


sns.pairplot(remove_outliers(ALL_TRAIN_DATAFRAME[CORE_NUMERICAL_COLUMNS],
                             NUMERICAL_QUANTILES))


# In[11]:


sns.pairplot(remove_outliers(TEST_DATAFRAME[CORE_NUMERICAL_COLUMNS],
                             NUMERICAL_QUANTILES))


# In[12]:


TRAIN_DATAFRAME = remove_outliers(TRAIN_DATAFRAME, NUMERICAL_QUANTILES)


# In[13]:


TRAIN_DATAFRAME.head()


# In[14]:


normalized_categories = sorted(normalize_categories(categories_from_column(TRAIN_DATAFRAME, 'features')))
normalized_categories[:50]


# In[15]:


most_common_ngrams = sorted(count_ngrams_up_to_n(" ".join(normalized_categories), 3).most_common(),
                            key=lambda x: (-x[1], x[0]))
most_common_ngrams[:50]


# In[16]:


most_common_ngrams = sorted(list(remove_small_or_stopwords_from_ranking(most_common_ngrams, nlp, 3)),
                            key=lambda x: (-x[1], x[0]))
most_common_ngrams[:50]


# In[17]:


TRAIN_DATAFRAME = column_list_to_category_flags(TRAIN_DATAFRAME, 'features', list(map(operator.itemgetter(0), most_common_ngrams[:100])))
VALIDATION_DATAFRAME = column_list_to_category_flags(VALIDATION_DATAFRAME, 'features', list(map(operator.itemgetter(0), most_common_ngrams[:100])))
TEST_DATAFRAME = column_list_to_category_flags(TEST_DATAFRAME, 'features', list(map(operator.itemgetter(0), most_common_ngrams[:100])))


# In[18]:


TRAIN_DATAFRAME.head(5)


# In[19]:


TRAIN_DATAFRAME = remap_date_column_to_days_before(TRAIN_DATAFRAME, "created", "created_days_ago", datetime.datetime(2017, 1, 1))
VALIDATION_DATAFRAME = remap_date_column_to_days_before(VALIDATION_DATAFRAME, "created", "created_days_ago", datetime.datetime(2017, 1, 1))
TEST_DATAFRAME = remap_date_column_to_days_before(TEST_DATAFRAME, "created", "created_days_ago", datetime.datetime(2017, 1, 1))


# In[20]:


TRAIN_DATAFRAME["created_days_ago"].head(5)


# In[21]:


INTEREST_LEVEL_MAPPINGS = {
    "high": 0,
    "medium": 1,
    "low": 2
}

TRAIN_DATAFRAME = remap_column(TRAIN_DATAFRAME, "interest_level", "label_interest_level", lambda x: INTEREST_LEVEL_MAPPINGS[x])
VALIDATION_DATAFRAME = remap_column(VALIDATION_DATAFRAME, "interest_level", "label_interest_level", lambda x: INTEREST_LEVEL_MAPPINGS[x])
# The TEST_DATAFRAME does not have an interest_level column, so we
# instead add it and replace it with all zeros
TEST_DATAFRAME["label_interest_level"] = 0


# In[22]:


TRAIN_DATAFRAME["label_interest_level"].head(5)


# In[23]:


((BUILDING_ID_UNKNOWN_REMAPPING,
  BUILDING_CATEGORY_TO_BUILDING_ID,
  BUILDING_CATEGORY_TO_BUILDING_ID),
 (TRAIN_DATAFRAME,
  VALIDATION_DATAFRAME,
  TEST_DATAFRAME)) = map_categorical_column_to_category_ids(
    'building_id',
    'building_id_category',
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
    min_freq=40
)


# In[24]:


((MANAGER_ID_UNKNOWN_REMAPPING,
  MANAGER_ID_TO_MANAGER_CATEGORY,
  MANAGER_CATEGORY_TO_MANAGER_ID),
 (TRAIN_DATAFRAME,
  VALIDATION_DATAFRAME,
  TEST_DATAFRAME)) = map_categorical_column_to_category_ids(
    'manager_id',
    'manager_id_category',
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
    min_freq=40
)


# In[25]:


(TRAIN_DATAFRAME,
 VALIDATION_DATAFRAME,
 TEST_DATAFRAME) = parse_address_components(
    [
        "display_address",
        "street_address"
    ],
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
)


# In[26]:


((DISP_ADDR_ID_UNKNOWN_REMAPPING,
  DISP_ADDR_ID_TO_DISP_ADDR_CATEGORY,
  DISP_ADDR_CATEGORY_TO_DISP_ADDR_ID),
 (TRAIN_DATAFRAME,
  VALIDATION_DATAFRAME,
  TEST_DATAFRAME)) = map_categorical_column_to_category_ids(
    'display_address_normalized',
    'display_address_category',
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
    min_freq=40
)


# In[27]:


(TRAIN_DATAFRAME,
 VALIDATION_DATAFRAME,
 TEST_DATAFRAME) = count_json_in_dataframes(
    "photos",
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
)


# In[28]:


NUMERICAL_COLUMNS = CORE_NUMERICAL_COLUMNS + [
    'photos_count'
]


# In[29]:


(TRAIN_DATAFRAME,
 VALIDATION_DATAFRAME,
 TEST_DATAFRAME) = remap_columns_with_transform(
    'description',
    'clean_description',
    normalize_description,
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME,
)


# In[30]:


DROP_COLUMNS = [
    'id',
    'index',
    'created',
    'building_id',
    'clean_description',
    'description',
    'features',
    'display_address',
    'display_address_normalized',
    # We keep listing_id in the dataframe
    # since we'll need it later
    # 'listing_id',
    'manager_id',
    'photos',
    'street_address',
    'street_address_normalized',
    'interest_level',
]


# In[31]:


(FEATURES_TRAIN_DATAFRAME,
 FEATURES_VALIDATION_DATAFRAME,
 FEATURES_TEST_DATAFRAME) = drop_columns_from_dataframes(
    DROP_COLUMNS,
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME
)


# In[32]:


FEATURES_TRAIN_DATAFRAME.head(5)


# In[33]:


FEATURIZED_NUMERICAL_COLUMNS = CORE_NUMERICAL_COLUMNS + ["photos_count", "label_interest_level"]


# In[34]:


sns.pairplot(remove_outliers(FEATURES_TRAIN_DATAFRAME[FEATURIZED_NUMERICAL_COLUMNS],
                             NUMERICAL_QUANTILES))


# In[35]:


sns.pairplot(remove_outliers(FEATURES_VALIDATION_DATAFRAME[FEATURIZED_NUMERICAL_COLUMNS],
                             NUMERICAL_QUANTILES))


# In[36]:


sns.pairplot(remove_outliers(FEATURES_TEST_DATAFRAME[FEATURIZED_NUMERICAL_COLUMNS[:-1]],
                             NUMERICAL_QUANTILES))


# In[37]:


CATEGORICAL_FEATURES = {
    'building_id_category': len(BUILDING_CATEGORY_TO_BUILDING_ID),
    'manager_id_category': len(MANAGER_ID_TO_MANAGER_CATEGORY),
    'display_address_category': len(DISP_ADDR_ID_TO_DISP_ADDR_CATEGORY)
}


# In[38]:


TRAIN_LABELS = FEATURES_TRAIN_DATAFRAME['label_interest_level']
VALIDATION_LABELS = FEATURES_VALIDATION_DATAFRAME['label_interest_level']


# In[39]:


def train_logistic_regression_model(data_info,
                                    featurized_train_data,
                                    featurized_validation_data,
                                    train_labels,
                                    validation_labels,
                                    train_param_grid_optimal=None):
    pipeline = basic_logistic_regression_pipeline(featurized_train_data,
                                                  train_labels,
                                                  CATEGORICAL_FEATURES,
                                                  param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


def predict_with_sklearn_estimator(model, data):
    return model.predict(data), model.predict_proba(data)


# In[40]:


(LOGISTIC_REGRESSION_MODEL_VALIDATION_PROBABILITIES,
 LOGISTIC_REGRESSION_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tabular_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_logistic_regression_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'C': [1.0],
            'class_weight': [None],
            'penalty': ['l2']
        }
    )
)


# In[41]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        LOGISTIC_REGRESSION_MODEL_TEST_PROBABILITIES,
    ),
    'renthop_logistic_regression_submissions.csv'
)


# In[42]:


def train_xgboost_model(data_info,
                        featurized_train_data,
                        featurized_validation_data,
                        train_labels,
                        validation_labels,
                        train_param_grid_optimal=None):
    pipeline = basic_xgboost_pipeline(featurized_train_data,
                                      train_labels,
                                      tree_method=(
                                          # 'gpu_hist' turned out to be a lot slower
                                         'hist'
                                      ),
                                      param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


# In[43]:


(XGBOOST_MODEL_VALIDATION_PROBABILITIES,
 XGBOOST_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tree_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_xgboost_model,
        predict_with_sklearn_estimator,
        # Determined by Grid Search, above
        train_param_grid_optimal={
            'colsample_bytree': [1.0],
            'gamma': [1.5],
            'max_depth': [5],
            'min_child_weight': [1],
            'n_estimators': [200],
            'subsample': [0.6]
        }
    )
)


# In[44]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        XGBOOST_MODEL_TEST_PROBABILITIES,
    ),
    'xgboost_submissions.csv'
)


# In[45]:


def train_rf_model(data_info,
                   featurized_train_data,
                   featurized_validation_data,
                   train_labels,
                   validation_labels,
                   train_param_grid_optimal=None):
    pipeline = basic_random_forest_pipeline(featurized_train_data,
                                            train_labels,
                                            # Determined by Grid Search, above
                                            param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


# In[46]:


(RF_MODEL_VALIDATION_PROBABILITIES,
 RF_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tree_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_rf_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'bootstrap': [False],
            'max_depth': [5],
            'min_samples_leaf': [1],
            'n_estimators': [100]
        }
    )
)


# In[47]:


def train_adaboost_model(data_info,
                         featurized_train_data,
                         featurized_validation_data,
                         train_labels,
                         validation_labels,
                         train_param_grid_optimal=None):
    pipeline = basic_adaboost_pipeline(featurized_train_data,
                                       train_labels,
                                       # Determined by Grid Search, above
                                       param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


# In[48]:


(ADABOOST_MODEL_VALIDATION_PROBABILITIES,
 ADABOOST_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tree_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_adaboost_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'learning_rate': [0.1],
            'n_estimators': [100]
        }
    )
)


# In[49]:


def train_extratrees_model(data_info,
                           featurized_train_data,
                           featurized_validation_data,
                           train_labels,
                           validation_labels,
                           train_param_grid_optimal=None):
    pipeline = basic_extratrees_pipeline(featurized_train_data,
                                         train_labels,
                                         # Determined by Grid Search, above
                                         param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


# In[50]:


(EXTRATREES_MODEL_VALIDATION_PROBABILITIES,
 EXTRATREES_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tree_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_extratrees_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'bootstrap': [False],
            'max_depth': [5],
            'min_samples_leaf': [1],
            'n_estimators': [200]
        }
    )
)


# In[51]:


def train_svc_model(data_info,
                    featurized_train_data,
                    featurized_validation_data,
                    train_labels,
                    validation_labels,
                    train_param_grid_optimal=None):
    pipeline = basic_svc_pipeline(featurized_train_data,
                                  train_labels,
                                  # Determined by Grid Search, above
                                  param_grid_optimal=train_param_grid_optimal)
    pipeline.fit(featurized_train_data, train_labels)
    print("Best parameters {}".format(pipeline.best_params_))
    return pipeline


# In[52]:


(SVC_MODEL_VALIDATION_PROBABILITIES,
 SVC_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_for_tabular_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_svc_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'C': [1.0],
            'gamma': ['scale'],
            'kernel': ['rbf']
        }
    )
)


# In[53]:


((TRAIN_FEATURES_CONTINUOUS,
  TRAIN_FEATURES_CATEGORICAL),
 (VALIDATION_FEATURES_CONTINUOUS,
  VALIDATION_FEATURES_CATEGORICAL),
 (TEST_FEATURES_CONTINUOUS,
  TEST_FEATURES_CATEGORICAL)) = rescale_features_and_split_into_continuous_and_categorical(CATEGORICAL_FEATURES,
                                                                                           FEATURES_TRAIN_DATAFRAME,
                                                                                           FEATURES_VALIDATION_DATAFRAME,
                                                                                           FEATURES_TEST_DATAFRAME)


# In[54]:


TRAIN_LABELS_TENSOR = torch.tensor(TRAIN_LABELS.values).long()
VALIDATION_LABELS_TENSOR = torch.tensor(VALIDATION_LABELS.values).long()


# In[55]:


TRAIN_FEATURES_CONTINUOUS_TENSOR = torch.tensor(TRAIN_FEATURES_CONTINUOUS).float()
TRAIN_FEATURES_CATEGORICAL_TENSOR = torch.tensor(TRAIN_FEATURES_CATEGORICAL).long()


# In[56]:


VALIDATION_FEATURES_CONTINUOUS_TENSOR = torch.tensor(VALIDATION_FEATURES_CONTINUOUS).float()
VALIDATION_FEATURES_CATEGORICAL_TENSOR = torch.tensor(VALIDATION_FEATURES_CATEGORICAL).long()


# In[57]:


TEST_FEATURES_CONTINUOUS_TENSOR = torch.tensor(TEST_FEATURES_CONTINUOUS).float()
TEST_FEATURES_CATEGORICAL_TENSOR = torch.tensor(TEST_FEATURES_CATEGORICAL).long()


# In[58]:


preview_tokenization(TRAIN_DATAFRAME["description"][:10])


# In[59]:


preview_encoded_sentences(TRAIN_DATAFRAME["description"][:10])


# In[60]:


def featurize_for_rnn_language_model(*dataframes):
    data_info, model_datasets = featurize_sequences_from_dataframe(*dataframes)
    return data_info, model_datasets


def train_rnn_model(data_info,
                    featurized_train_data,
                    featurized_validation_data,
                    train_labels,
                    validation_labels,
                    train_param_grid_optimal=None):
    word_to_one_hot, one_hot_to_word = data_info
    train_word_description_sequences, train_word_sequences_lengths = featurized_train_data
    model = NoToTensorInLossClassifier(
        SimpleRNNPredictor,
        module__encoder_dimension=100, # Number of encoder features
        module__hidden_dimension=50, # Number of hidden features
        module__dictionary_dimension=len(one_hot_to_word), # Dictionary dimension
        module__output_dimension=3,
        module__dropout=0.1,
        lr=1e-2,
        batch_size=256,
        optimizer=optim.Adam,
        max_epochs=4,
        module__layers=2,
        train_split=simple_train_test_split_without_shuffle_func(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='rnn_lang_checkpoint'),
            TrainEndCheckpoint(dirname='rnn_lang_checkpoint',
                               fn_prefix='rnn_train_end_')
        ]
    )
    model.fit((train_word_description_sequences,
               train_word_sequences_lengths),
              maybe_cuda(train_labels))
    
    return model


# In[61]:


(SIMPLE_RNN_MODEL_VALIDATION_PROBABILITIES,
 SIMPLE_RNN_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_for_rnn_language_model,
        train_rnn_model,
        predict_with_sklearn_estimator
    )
)


# In[62]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        SIMPLE_RNN_MODEL_TEST_PROBABILITIES,
    ),
    'simple_rnn_model_submissions.csv'
)


# In[63]:


def splice_into_datasets(datasets, append):
    return tuple(list(d) + list(a) for d, a in zip(datasets, append))

def featurize_for_rnn_tabular_model(*dataframes):
    data_info, model_datasets = featurize_sequences_from_dataframe(*dataframes)
    # Need to wrap each dataset in outer list so that splicing works correctly
    return data_info, splice_into_datasets(model_datasets,
                                           ((TRAIN_FEATURES_CONTINUOUS_TENSOR,
                                             TRAIN_FEATURES_CATEGORICAL_TENSOR),
                                            (VALIDATION_FEATURES_CONTINUOUS_TENSOR,
                                             VALIDATION_FEATURES_CATEGORICAL_TENSOR),
                                            (TEST_FEATURES_CONTINUOUS_TENSOR,
                                             TEST_FEATURES_CATEGORICAL_TENSOR )))


def train_rnn_tabular_model(data_info,
                            featurized_train_data,
                            featurized_validation_data,
                            train_labels,
                            validation_labels,
                            train_param_grid_optimal=None):
    word_to_one_hot, one_hot_to_word = data_info
    _, _, train_continuous, train_categorical = featurized_train_data
    model = NoToTensorInLossClassifier(
        SimpleRNNTabularDataPredictor,
        module__encoder_dimension=100, # Number of encoder features
        module__hidden_dimension=50, # Number of hidden features
        module__dictionary_dimension=len(one_hot_to_word), # Dictionary dimension
        module__output_dimension=3,
        module__dropout=0.1,
        module__continuous_features_dimension=train_continuous.shape[1],
        module__categorical_feature_embedding_dimensions=[
            (CATEGORICAL_FEATURES[c], 80) for c in CATEGORICAL_FEATURES
        ],
        lr=1e-2,
        batch_size=256,
        optimizer=optim.Adam,
        max_epochs=4,
        module__layers=2,
        train_split=simple_train_test_split_without_shuffle_func(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='rnn_lang_checkpoint'),
            TrainEndCheckpoint(dirname='rnn_lang_checkpoint',
                               fn_prefix='rnn_train_end_')
        ]
    )
    model.fit(featurized_train_data, maybe_cuda(train_labels))
    
    return model


# In[64]:


(SIMPLE_RNN_TABULAR_MODEL_VALIDATION_PROBABILITIES,
 SIMPLE_RNN_TABULAR_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_for_rnn_tabular_model,
        train_rnn_tabular_model,
        predict_with_sklearn_estimator
    )
)


# In[65]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        SIMPLE_RNN_TABULAR_MODEL_TEST_PROBABILITIES,
    ),
    'simple_rnn_model_tabular_data_submissions.csv'
)


# In[66]:


def featurize_dataframe_sequences_for_fasttext(*dataframes):
    sequences_lists = tokenize_sequences(
        *tuple(list(df['clean_description']) for df in dataframes)
    )

    text = torchtext_create_text_vocab(*sequences_lists,
                                       vectors='glove.6B.100d')
    
    return text, torchtext_process_texts(*postprocess_sequences(
        *sequences_lists,
        postprocessing=generate_bigrams
    ), text=text)


def train_fasttext_model(data_info,
                         featurized_train_data,
                         featurized_validation_data,
                         train_labels,
                         validation_labels,
                         train_param_grid_optimal=None):
    embedding_dim = 100
    model = NoToTensorInLossClassifier(
        FastText,
        lr=0.001,
        batch_size=256,
        optimizer=optim.Adam,
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='fasttext_checkpoint'),
            TrainEndCheckpoint(dirname='fasttext_tabular_checkpoint',
                               fn_prefix='fasttext_train_end_')
        ],
        max_epochs=6,
        train_split=shuffled_train_test_split_by_indices(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        module__encoder_dimension=embedding_dim, # Number of encoder features
        module__dictionary_dimension=len(data_info.vocab.itos), # Dictionary dimension
        module__output_dimension=3,
        module__dropout=0.8,
        module__pretrained=data_info.vocab.vectors
    )
    model.fit(featurized_train_data, maybe_cuda(train_labels))
    return model


# In[67]:


(FASTTEXT_MODEL_VALIDATION_PROBABILITIES,
 FASTTEXT_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_dataframe_sequences_for_fasttext,
        train_fasttext_model,
        predict_with_sklearn_estimator
    )
)


# In[68]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        FASTTEXT_MODEL_TEST_PROBABILITIES,
    ),
    'fasttext_model_submissions.csv'
)


# In[69]:


def featurize_for_fasttext_tabular_model(*dataframes):
    data_info, model_datasets = featurize_dataframe_sequences_for_fasttext(*dataframes)
    # Need to wrap each dataset in outer list so that splicing works correctly
    return data_info, splice_into_datasets(tuple([x] for x in model_datasets),
                                           ((TRAIN_FEATURES_CONTINUOUS_TENSOR,
                                             TRAIN_FEATURES_CATEGORICAL_TENSOR),
                                            (VALIDATION_FEATURES_CONTINUOUS_TENSOR,
                                             VALIDATION_FEATURES_CATEGORICAL_TENSOR),
                                            (TEST_FEATURES_CONTINUOUS_TENSOR,
                                             TEST_FEATURES_CATEGORICAL_TENSOR)))


def train_fasttext_tabular_model(data_info,
                                 featurized_train_data,
                                 featurized_validation_data,
                                 train_labels,
                                 validation_labels,
                                 train_param_grid_optimal=None):
    embedding_dim = 100
    _, train_features_continuous, _ = featurized_train_data
    model = NoToTensorInLossClassifier(
        FastTextWithTabularData,
        lr=0.001,
        batch_size=256,
        optimizer=optim.Adam,
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='fasttext_checkpoint'),
            TrainEndCheckpoint(dirname='fasttext_tabular_checkpoint',
                               fn_prefix='fasttext_train_end_')
        ],
        max_epochs=6,
        train_split=shuffled_train_test_split_by_indices(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        module__encoder_dimension=embedding_dim, # Number of encoder features
        module__dictionary_dimension=len(data_info.vocab.itos), # Dictionary dimension
        module__output_dimension=3,
        module__dropout=0.8,
        module__pretrained=data_info.vocab.vectors,
        module__continuous_features_dimension=train_features_continuous.shape[1],
        module__categorical_feature_embedding_dimensions=[
            (CATEGORICAL_FEATURES[c], 80) for c in CATEGORICAL_FEATURES
        ],
    )
    model.fit(featurized_train_data, maybe_cuda(train_labels))
    return model


# In[70]:


(FASTTEXT_TABULAR_MODEL_VALIDATION_PROBABILITIES,
 FASTTEXT_TABULAR_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_for_fasttext_tabular_model,
        train_fasttext_tabular_model,
        predict_with_sklearn_estimator
    )
)


# In[71]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        FASTTEXT_TABULAR_MODEL_TEST_PROBABILITIES,
    ),
    'fasttext_tabular_model_submissions.csv'
)


# In[72]:


def featurize_dataframe_sequences_for_textcnn(*dataframes):
    sequences_lists = tokenize_sequences(
        *tuple(list(df['clean_description']) for df in dataframes)
    )

    text = torchtext_create_text_vocab(*sequences_lists,
                                       vectors='glove.6B.100d')
    
    return text, tuple(text.process(sl).transpose(0, 1) for sl in sequences_lists)

def train_textcnn_model(data_info,
                        featurized_train_data,
                        featurized_validation_data,
                        train_labels,
                        validation_labels,
                        train_param_grid_optimal=None):
    embedding_dim = 100
    model = NoToTensorInLossClassifier(
        TextCNN,
        lr=0.001,
        batch_size=64,
        optimizer=optim.Adam,
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='textcnn_checkpoint'),
            TrainEndCheckpoint(dirname='textcnn_tabular_checkpoint',
                               fn_prefix='textcnn_train_end_')
        ],
        max_epochs=10,
        train_split=shuffled_train_test_split_by_indices(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        module__encoder_dimension=embedding_dim, # Number of encoder features
        module__dictionary_dimension=len(data_info.vocab.itos), # Dictionary dimension
        module__output_dimension=3,
        module__n_filters=10,
        module__filter_sizes=(3, 4, 5),
        module__dropout=0.8,
        module__pretrained=data_info.vocab.vectors
    )
    model.fit(featurized_train_data, maybe_cuda(train_labels))
    return model


# In[73]:


(TEXTCNN_MODEL_VALIDATION_PROBABILITIES,
 TEXTCNN_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_dataframe_sequences_for_textcnn,
        train_textcnn_model,
        predict_with_sklearn_estimator
    )
)


# In[74]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        TEXTCNN_MODEL_TEST_PROBABILITIES,
    ),
    'textcnn_model_submissions.csv'
)


# In[75]:


def featurize_dataframes_for_textcnn_tabular_model(*dataframes):
    data_info, model_datasets = featurize_dataframe_sequences_for_textcnn(*dataframes)
    return data_info, splice_into_datasets(tuple([x] for x in model_datasets),
                                           ((TRAIN_FEATURES_CONTINUOUS_TENSOR,
                                             TRAIN_FEATURES_CATEGORICAL_TENSOR),
                                            (VALIDATION_FEATURES_CONTINUOUS_TENSOR,
                                             VALIDATION_FEATURES_CATEGORICAL_TENSOR),
                                            (TEST_FEATURES_CONTINUOUS_TENSOR,
                                             TEST_FEATURES_CATEGORICAL_TENSOR)))

def train_textcnn_tabular_model(data_info,
                                featurized_train_data,
                                featurized_validation_data,
                                train_labels,
                                validation_labels,
                                train_param_grid_optimal=None):
    embedding_dim = 100
    _, train_features_continuous, _ = featurized_train_data
    model = NoToTensorInLossClassifier(
        TextCNNWithTabularData,
        lr=0.001,
        batch_size=256,
        optimizer=optim.Adam,
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            LRAnnealing(),
            LRScheduler(),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='fasttext_checkpoint'),
            TrainEndCheckpoint(dirname='fasttext_tabular_checkpoint',
                               fn_prefix='fasttext_train_end_')
        ],
        max_epochs=10,
        train_split=shuffled_train_test_split_by_indices(0.3),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        module__encoder_dimension=embedding_dim, # Number of encoder features
        module__dictionary_dimension=len(data_info.vocab.itos), # Dictionary dimension
        module__output_dimension=3,
        module__n_filters=100,
        module__filter_sizes=(3, 4, 5),
        module__dropout=0.8,
        module__pretrained=data_info.vocab.vectors,
        module__continuous_features_dimension=train_features_continuous.shape[1],
        module__categorical_feature_embedding_dimensions=[
            (CATEGORICAL_FEATURES[c], 80) for c in CATEGORICAL_FEATURES
        ],
    )
    model.fit(featurized_train_data, maybe_cuda(train_labels))
    return model


# In[76]:


(TEXTCNN_TABULAR_MODEL_VALIDATION_PROBABILITIES,
 TEXTCNN_TABULAR_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_dataframes_for_textcnn_tabular_model,
        train_textcnn_tabular_model,
        predict_with_sklearn_estimator
    )
)


# In[77]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        TEXTCNN_TABULAR_MODEL_TEST_PROBABILITIES,
    ),
    'textcnn_tabular_model_submissions.csv'
)


# In[78]:


gc_and_clear_caches(None)


# In[79]:


def flatten_params(params):
    return {
        k: v[0] if isinstance(v, list) else v for k, v in params.items()
    }


# In[80]:


BERT_MODEL = 'bert-base-uncased'


def featurize_bert_lang_features(*dataframes):
    return _, tuple(
        tuple(torch.stack(x) for x in zip(*features))
        for features in bert_featurize_data_frames(BERT_MODEL, *dataframes)
    )
        

def train_bert_lang_model(data_info,
                          featurized_train_data,
                          featurized_validation_data,
                          train_labels,
                          validation_labels,
                          train_param_grid_optimal=None):
    model = BertClassifier(
        module=create_bert_model(BERT_MODEL, 3),
        optimizer__warmup=WARMUP_PROPORTION,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer=BertAdam,
        lr=6e-5,
        len_train_data=int(len(featurized_train_data[0])),
        num_labels=3,
        batch_size=16,
        train_split=shuffled_train_test_split_by_indices(0.1),
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='bert_lang_checkpoint')
        ],
    )
    
    if not train_param_grid_optimal:
        # As sugested by Maksad, need to do a hyperparameter search
        # here to get good results.
        param_grid = {
            "batch_size": [16, 32],
            "lr": [6e-5, 3e-5, 3e-1, 2e-5],
            "max_epochs": [3, 4]
        }
        search = GridSearchCV(model,
                              param_grid,
                              cv=1,
                              refit=False,
                              scoring=make_scorer(log_loss,
                                                  greater_is_better=False,
                                                  needs_proba=True))
        search.fit(TensorTuple(featurized_train_data), train_labels)

        print('Best params {}'.format(search.best_params_))
        # Now re-fit the estimator manually, using the best params -
        # we do this manually since we need a different view over
        # the training data to make it work
        best = clone(search.estimator, safe=True).set_params(**search.best_params_)
        best.fit(featurized_train_data, train_labels)
        return best
    else:
        model = clone(model, safe=True).set_params(**flatten_params(train_param_grid_optimal))
        model.fit(featurized_train_data, train_labels)
        return model


# In[81]:


(BERT_LANG_MODEL_VALIDATION_PROBABILITIES,
 BERT_LANG_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS,
        VALIDATION_LABELS,
        featurize_bert_lang_features,
        train_bert_lang_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'lr': [2e-05],
            'max_epochs': [4],
            'batch_size': [32]
        }
    )
)


# In[82]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        BERT_LANG_MODEL_TEST_PROBABILITIES,
    ),
    'bert_model_submissions.csv'
)


# In[83]:


def featurize_bert_tabular_features(*dataframes):
    model_datasets = splice_into_datasets(tuple([x] for x in bert_featurize_data_frames(BERT_MODEL, *dataframes)),
                                          ((TRAIN_FEATURES_CONTINUOUS_TENSOR,
                                            TRAIN_FEATURES_CATEGORICAL_TENSOR),
                                           (VALIDATION_FEATURES_CONTINUOUS_TENSOR,
                                            VALIDATION_FEATURES_CATEGORICAL_TENSOR),
                                           (TEST_FEATURES_CONTINUOUS_TENSOR,
                                            TEST_FEATURES_CATEGORICAL_TENSOR)))
    return _, tuple(
        tuple(torch.stack(x) for x in zip(*[
            tuple(list(bert_features) + [continuous, categorical])
            for bert_features, continuous, categorical in zip(features,
                                                              continuous_tensor,
                                                              categorical_tensor)
        ]))
        for features, continuous_tensor, categorical_tensor in model_datasets
    )


def train_bert_tabular_model(data_info,
                             featurized_train_data,
                             featurized_validation_data,
                             train_labels,
                             validation_labels,
                             train_param_grid_optimal=None):
    batch_size = 16
    _, _, _, continuous_features, categorical_features = featurized_train_data
    model = BertClassifier(
        module=create_bert_model_with_tabular_features(
            BERT_MODEL,
            continuous_features.shape[1],
            [
                (CATEGORICAL_FEATURES[c], 80) for c in CATEGORICAL_FEATURES
            ],
            3
        ),
        len_train_data=int(len(featurized_train_data[0])),
        optimizer__warmup=WARMUP_PROPORTION,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        optimizer=BertAdam,
        num_labels=3,
        batch_size=batch_size,
        train_split=shuffled_train_test_split_by_indices(0.3),
        callbacks=[
            SummarizeParameters(),
            EpochScoring(scoring='accuracy'),
            ProgressBar(),
            CheckpointAndKeepBest(dirname='bert_lang_checkpoint')
        ],
    )
    
    if not train_param_grid_optimal:
        # As sugested by Maksad, need to do a hyperparameter search
        # here to get good results.
        param_grid = {
            "batch_size": [16, 32],
            "lr": [6e-5, 3e-5, 3e-5, 2e-5],
            "max_epochs": [3, 4]
        }
        search = GridSearchCV(model,
                              param_grid,
                              cv=2,
                              refit=False,
                              scoring=make_scorer(log_loss,
                                                  greater_is_better=False,
                                                  needs_proba=True))
        search.fit(TensorTuple(featurized_train_data), train_labels)

        print('Best params {}'.format(search.best_params_))
        # Now re-fit the estimator manually, using the best params -
        # we do this manually since we need a different view over
        # the training data to make it work
        best = clone(search.estimator, safe=True).set_params(**search.best_params_)
        best.fit(featurized_train_data, train_labels)
        return best
    else:
        model = clone(model, safe=True).set_params(**flatten_params(train_param_grid_optimal))
        model.fit(featurized_train_data, train_labels)
        return model


# In[84]:


(BERT_TABULAR_MODEL_VALIDATION_PROBABILITIES,
 BERT_TABULAR_MODEL_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        TRAIN_DATAFRAME,
        VALIDATION_DATAFRAME,
        TEST_DATAFRAME,
        VALIDATION_DATAFRAME,
        TRAIN_LABELS_TENSOR,
        VALIDATION_LABELS_TENSOR,
        featurize_bert_tabular_features,
        train_bert_tabular_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'lr': [2e-05],
            'max_epochs': [4],
            'batch_size': [32]
        }
    )
)


# In[85]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        BERT_TABULAR_MODEL_TEST_PROBABILITIES,
    ),
    'bert_tabular_model_submissions.csv'
)


# In[86]:


from utils.language_models.ulmfit import train_ulmfit_model_and_get_validation_and_test_set_predictions

(ULMFIT_VALIDATION_PROBABILITIES,
 ULMFIT_TEST_PROBABILITIES) = train_ulmfit_model_and_get_validation_and_test_set_predictions(
    TRAIN_DATAFRAME,
    VALIDATION_DATAFRAME,
    TEST_DATAFRAME
)


# In[87]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        ULMFIT_TEST_PROBABILITIES
    ),
    'ulmfit_submissions.csv'
)


# In[88]:


(STACKED_VALIDATION_PREDICTIONS_TRAINING_SET,
 STACKED_VALIDATION_PREDICTIONS_VALIDATION_SET,
 _,
 VALIDATION_SPLIT_VALIDATION_DATAFRAME,
 TRAINING_SPLIT_FEATURES_VALIDATION_DATAFRAME,
 VALIDATION_SPLIT_FEATURES_VALIDATION_DATAFRAME,
 STACKED_VALIDATION_PREDICTIONS_LABELS_TRAINING_SET,
 STACKED_VALIDATION_PREDICTIONS_LABELS_VALIDATION_SET) = train_test_split(
    np.column_stack([
        LOGISTIC_REGRESSION_MODEL_VALIDATION_PROBABILITIES,
        XGBOOST_MODEL_VALIDATION_PROBABILITIES,
        RF_MODEL_VALIDATION_PROBABILITIES,
        ADABOOST_MODEL_VALIDATION_PROBABILITIES,
        EXTRATREES_MODEL_VALIDATION_PROBABILITIES,
        SVC_MODEL_VALIDATION_PROBABILITIES,
        SIMPLE_RNN_MODEL_VALIDATION_PROBABILITIES,
        SIMPLE_RNN_TABULAR_MODEL_VALIDATION_PROBABILITIES,
        FASTTEXT_MODEL_VALIDATION_PROBABILITIES,
        FASTTEXT_TABULAR_MODEL_VALIDATION_PROBABILITIES,
        TEXTCNN_MODEL_VALIDATION_PROBABILITIES,
        TEXTCNN_TABULAR_MODEL_VALIDATION_PROBABILITIES,
        BERT_LANG_MODEL_VALIDATION_PROBABILITIES,
        BERT_TABULAR_MODEL_VALIDATION_PROBABILITIES,
        ULMFIT_VALIDATION_PROBABILITIES,
    ]),
    VALIDATION_DATAFRAME,
    FEATURES_VALIDATION_DATAFRAME,
    VALIDATION_LABELS,
    stratify=VALIDATION_LABELS,
    test_size=0.1
)

STACKED_TEST_PREDICTIONS_TEST_SET = np.column_stack([
    LOGISTIC_REGRESSION_MODEL_TEST_PROBABILITIES,
    XGBOOST_MODEL_TEST_PROBABILITIES,
    RF_MODEL_TEST_PROBABILITIES,
    ADABOOST_MODEL_TEST_PROBABILITIES,
    EXTRATREES_MODEL_TEST_PROBABILITIES,
    SVC_MODEL_TEST_PROBABILITIES,
    SIMPLE_RNN_MODEL_TEST_PROBABILITIES,
    SIMPLE_RNN_TABULAR_MODEL_TEST_PROBABILITIES,
    FASTTEXT_MODEL_TEST_PROBABILITIES,
    FASTTEXT_TABULAR_MODEL_TEST_PROBABILITIES,
    TEXTCNN_MODEL_TEST_PROBABILITIES,
    TEXTCNN_TABULAR_MODEL_TEST_PROBABILITIES,
    BERT_LANG_MODEL_TEST_PROBABILITIES,
    BERT_TABULAR_MODEL_TEST_PROBABILITIES,
    ULMFIT_TEST_PROBABILITIES,
])


# In[89]:


def identity_unpack(*args):
    return _, args

(XGBOOST_MODEL_STACKED_VALIDATION_PROBABILITIES,
 XGBOOST_MODEL_STACKED_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_TRAINING_SET),
        pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_VALIDATION_SET),
        pd.DataFrame(STACKED_TEST_PREDICTIONS_TEST_SET),
        VALIDATION_SPLIT_VALIDATION_DATAFRAME,
        STACKED_VALIDATION_PREDICTIONS_LABELS_TRAINING_SET.reset_index(drop=True),
        STACKED_VALIDATION_PREDICTIONS_LABELS_VALIDATION_SET.reset_index(drop=True),
        identity_unpack,
        train_xgboost_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'colsample_bytree': [0.8],
            'gamma': [2],
            'max_depth': [3],
            'min_child_weight': [5],
            'n_estimators': [100],
            'subsample': [0.8]
        }
    )
)


# In[90]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        XGBOOST_MODEL_STACKED_TEST_PROBABILITIES
    ),
    'xgboost_stacked_submissions.csv'
)


# In[91]:


(LOGISTIC_REGRESSION_MODEL_STACKED_VALIDATION_PROBABILITIES,
 LOGISTIC_REGRESSION_MODEL_STACKED_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_TRAINING_SET),
        pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_VALIDATION_SET),
        pd.DataFrame(STACKED_TEST_PREDICTIONS_TEST_SET),
        VALIDATION_SPLIT_VALIDATION_DATAFRAME,
        STACKED_VALIDATION_PREDICTIONS_LABELS_TRAINING_SET.reset_index(drop=True),
        STACKED_VALIDATION_PREDICTIONS_LABELS_VALIDATION_SET.reset_index(drop=True),
        identity_unpack,
        train_logistic_regression_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'C': [1.0],
            'class_weight': [None],
            'penalty': ['l2']
        }
    )
)


# In[92]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        LOGISTIC_REGRESSION_MODEL_STACKED_TEST_PROBABILITIES
    ),
    'logistic_stacked_submissions.csv'
)


# In[93]:


(GUIDED_XGBOOST_MODEL_STACKED_VALIDATION_PROBABILITIES,
 GUIDED_XGBOOST_MODEL_STACKED_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        pd.concat((pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_TRAINING_SET),
                   TRAINING_SPLIT_FEATURES_VALIDATION_DATAFRAME.reset_index().drop(["index"], axis=1)), axis=1),
        pd.concat((pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_VALIDATION_SET),
                   VALIDATION_SPLIT_FEATURES_VALIDATION_DATAFRAME.reset_index().drop(["index"], axis=1)), axis=1),
        pd.concat((pd.DataFrame(STACKED_TEST_PREDICTIONS_TEST_SET),
                   FEATURES_TEST_DATAFRAME.reset_index().drop("index", axis=1)), axis=1),
        VALIDATION_SPLIT_VALIDATION_DATAFRAME,
        STACKED_VALIDATION_PREDICTIONS_LABELS_TRAINING_SET.reset_index(drop=True),
        STACKED_VALIDATION_PREDICTIONS_LABELS_VALIDATION_SET.reset_index(drop=True),
        featurize_for_tree_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_xgboost_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'colsample_bytree': [0.6],
            'gamma': [2],
            'max_depth': [3],
            'min_child_weight': [1],
            'n_estimators': [100],
            'subsample': [0.8]
        }
    )
)


# In[94]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        GUIDED_XGBOOST_MODEL_STACKED_TEST_PROBABILITIES
    ),
    'guided_xgboost_stacked_submissions.csv'
)


# In[95]:


(GUIDED_LOGISTIC_MODEL_STACKED_VALIDATION_PROBABILITIES,
 GUIDED_LOGISTIC_MODEL_STACKED_TEST_PROBABILITIES) = gc_and_clear_caches(
    train_model_and_get_validation_and_test_set_predictions(
        pd.concat((pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_TRAINING_SET),
                   TRAINING_SPLIT_FEATURES_VALIDATION_DATAFRAME.reset_index().drop(["index"], axis=1)), axis=1),
        pd.concat((pd.DataFrame(STACKED_VALIDATION_PREDICTIONS_VALIDATION_SET),
                   VALIDATION_SPLIT_FEATURES_VALIDATION_DATAFRAME.reset_index().drop(["index"], axis=1)), axis=1),
        pd.concat((pd.DataFrame(STACKED_TEST_PREDICTIONS_TEST_SET),
                   FEATURES_TEST_DATAFRAME.reset_index().drop("index", axis=1)), axis=1),
        VALIDATION_SPLIT_VALIDATION_DATAFRAME,
        STACKED_VALIDATION_PREDICTIONS_LABELS_TRAINING_SET.reset_index(drop=True),
        STACKED_VALIDATION_PREDICTIONS_LABELS_VALIDATION_SET.reset_index(drop=True),
        featurize_for_tabular_models(DROP_COLUMNS, CATEGORICAL_FEATURES),
        train_logistic_regression_model,
        predict_with_sklearn_estimator,
        train_param_grid_optimal={
            'C': [1.0],
            'class_weight': [None],
            'penalty': ['l2']
        }
    )
)


# In[96]:


write_predictions_table_to_csv(
    get_prediction_probabilities_with_columns_from_predictions(
        FEATURES_TEST_DATAFRAME['listing_id'],
        GUIDED_LOGISTIC_MODEL_STACKED_TEST_PROBABILITIES
    ),
    'guided_logistic_stacked_submissions.csv'
)

