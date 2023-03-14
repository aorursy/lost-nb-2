#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import re
from math import floor, ceil
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
import numpy as np
from torch.utils import data
from transformers import BertForMultipleChoice, BertTokenizer
import torch


RANDOM_SEED = 49662
SPLIT_NUM = 5


class QuestDataset(data.Dataset):

    def __init__(self, df, train_mode=True, labeled=True):
        """
        Load data file in `google-quest-challenge` folder, and set data into Pandas dataframe

        :param load: default 'train'
        train: Load `train.csv`
        test: Load `test.csv`
        """

        self.MAX_LEN = 512
        self.SEP_TOKEN_ID = 102
        self.input_cols = ['question_title', 'question_body', 'answer']
        self.target_cols = ['question_asker_intent_understanding', 'question_body_critical',
                            'question_conversational', 'question_expect_short_answer',
                            'question_fact_seeking', 'question_has_commonly_accepted_answer',
                            'question_interestingness_others', 'question_interestingness_self',
                            'question_multi_intent', 'question_not_really_a_question',
                            'question_opinion_seeking', 'question_type_choice',
                            'question_type_compare', 'question_type_consequence',
                            'question_type_definition', 'question_type_entity',
                            'question_type_instructions', 'question_type_procedure',
                            'question_type_reason_explanation', 'question_type_spelling',
                            'question_well_written', 'answer_helpful',
                            'answer_level_of_information', 'answer_plausible',
                            'answer_relevance', 'answer_satisfaction',
                            'answer_type_instructions', 'answer_type_procedure',
                            'answer_type_reason_explanation', 'answer_well_written']
        self.train_mode = train_mode
        self.labeled = labeled
        self.data = df
        self.tokenizer = BertTokenizer.from_pretrained('../input/pretrained-bert-models-for-pytorch/bert-base-uncased-vocab.txt')

    def __getitem__(self, index):
        row = self.data.iloc[index]
        token_ids, seg_ids = self.get_token_ids(row)
        if self.labeled:
            labels = self.get_label(row)
            return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids

    def __len__(self):
        return len(self.data)

    def cut_words(self, sentence, punc=True, entity=True):
        """
        Convert a string sentence into the list containing words and punctuations
        :param sentence:
        string sentence

        :param punc:
        preserve the punctuations in the sentence.
        True: preserve, False: not preserve

        :param entity:
        identify the entity e.g <formu> <url> <numb> ...
        True: open entity mode, False: close entity mode

        :return: list with words and punctuations
        """
        punc_regex = '\s[\(\):,\.\?/\'\";\[\]\{\}\-\=\+\_|\!@#%\^\&\*]+|[\(\):,\.\?/\'\";\[\]\{\}\-\=\+\_|\!@#%\^\&\*]+\s'
        formula_regex = '(?<!\w)~*\\\\*\${2}.+?\${2}(?!\w|\(|\{|\[|\\\\|!)|(?<!\w)~*\\\\*\${1}.+?\${1}(?!\w|\(|\{|\[|\\\\|!)|(?<!\w)~*\\\\*\${2}\n*(^.+?\n)+\n*\${2}(?!\w|\(|\{|\[|\\\\|!)'
        url_regex = '(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|](?=\s|$|\. |\.\n|\]|\))'
        js_regex = '(&lt; *script.*?&gt;)[\s\S]*?(&lt;/ *script *&gt;)'
        css_regex = '(&lt; *style.*?&gt;)[\s\S]*?(&lt;/ *style *&gt;)'
        php_regex = '&lt;\s*\?php([\w\W]+?)(\?&gt;|\n\n\n(?=[A-Z]))'
        html_regex = '&lt;\?{0,1}/{0,1}%{0,1}([a-zA-Z]+\s+)+([a-zA-Z,:%-\.]+\s*=(\\|\s){0,1}(\").*?(\")\s*)*\?{0,1}/{0,1}%{0,1}\s*[a-zA-Z]*\s*&gt;'
        html1_regex = '(?<!#include\s)(?<!#include\s\s)&lt;.*?&gt;'
        html2_regex = '\n *(&lt;([a-z]+) *[^/]*?&gt;) *\n'
        html3_regex = '&lt;\!\-\-[\w\W]*?\-\-&gt;'
        latex_regex = '\\\\document[\s\S]*?\\\\end\{document\}'
        formula1_regex = '\${2}[\s\S]+?\${2}'
        latex1_regex = '\\\\begin[\s\S]*?(\\\\end\{[a-zA-Z\*]+\}|\<latex\>)'
        latex2_regex = '[ \t%]*(\\\\[a-zA-Z\*]+)+(\[[a-zA-Z\d=,\n]*?\])*(\\{.*?\\}) *(?=\n|%).*'
        java_regex = '\n{1,6}((import .*?;)|(package .*?;))\n((( {4}){0,5}.*?(;|\{|\})\n)|(( {4}){0,5}//.*?\n)|(( {4}){0,5}/\**.*?\n)|(( {4}){0,5} \**.*?\n)|(( {4}){0,5}(public|@|class|if).*?\n)|\n)+'
        # c_regex = '\n{1,6}(( {2}){0,5}//.*?\n)*#include (&lt;|\").*?(&gt;|\")\n(#include (&lt;|\").*?(&gt;|\")\n)*\n*((( {2}){0,20}(int|void|double|bool|float|short|long|double|char) [a-zA-Z]+\(.*?\)(\s*\{|;)\s*)|(( {2}){0,20}.*?\{\s*)|(( {2}){0,20}.*?(\\|\)|/|;|,))\s*\n|(( {2}){0,20}[a-zA-Z]+.*?(;|\s*\{)(\s*//.*){0,1}\n)|(?<=\n)\s*\{\s*(?=\n)|(\#.*?\n)|\n|((( {2}){0,5})//.*?\n)|(( {2}){0,5}\}\s*\n))+'
        c_regex = '\n{1,6}(( {2}){0,5}//.*?\n)*#include (&lt;|\").*?(&gt;|\")\n(#include (&lt;|\").*?(&gt;|\")\n)*(using .*?;\n)*\n*[\d\D]*?\n{3}'
        c2_regex = '(#include (&lt;|\").*?(&gt;|\")\n)+[\s\S]*'
        java2_regex = '(((?<=\s)(public ){0,1}class .*?\{)|(using .*?;)|((?<=\s)(public ){0,1}class .*?\s+\{))\n((( {4}){0,5}.*?(;|\{|\})\n)|(( {4}){0,5}//.*?\n)|(( {4}){0,5}/\**.*?\n)|(( {4}){0,5} \**.*?\n)|(( {4}){0,5}(public|@|class|if).*?\n)|(( {4}){0,5}.*?\",\n)|\n)+'
        java3_regex = '(((?<=\s)(public |private |protected){0,1}.*?class .*?\s*\{)|((?<=\n)using .*?;))\n*((( {4}){0,5}.*?(;|\{|\})\s*\n)|(( {4}){0,5}//.*?\n)|(( {4}){0,5}/\**.*?\n)|(( {4}){0,5} \**.*?\n)|(( {4}){0,5}(public|@|class|if).*?\n)|(( {4}){0,5}.*?\",\n)|\n)+'
        java4_regex = '((?<=\n)(public |private |protected ){0,1}class .*?\s*(\{|extends))\n*((( {4}){0,5}.*?(;|\{|\})\s*\n)|(( {4}){0,5}//.*?\n)|(( {4}){0,5}/\**.*?\n)|(( {4}){0,5} \**.*?\n)|(( {4}){0,5}(public|@|class|if).*?\n)|(( {4}){0,5}.*?\",\n)|\n)+'
        java5_regex = '(\s*(private|public|protected) ((void|static) ){0,2}[a-zA-Z]+.*(\)|\}|;|\(|\s*\{)(?=\n)\n)+\n*((( {4}){0,5}.*?(;|\{|\})\s*\n)|(( {4}){0,5}//.*?\n)|(( {4}){0,5}/\**.*?\n)|(( {4}){0,5} \**.*?\n)|(( {4}){0,5}(public|@|class|if).*?\n)|(( {4}){0,5}.*?\",\n)|\n)+'
        python_regex = '(?<=\n) *((class [a-zA-Z_]+?\(.*?\))|(for [^\s]+ in .*?\:)|(if _{0,2}name_{0,2} \=\= \'_{0,2}main_{0,2}\'\:)|(def [^\s]+?\(.*?\))|((from [^\s]*? +){0,1}import.*))[\s\S]*?((\n\n(?=([A-Z][a-z]+|[a-z]+ ){2}([A-Z][a-z]+|[a-z]+ )+(?!\=)))|(\n\n(?=\n))|\n+$)'
        sql_regex = '(?<=\n)\n*(Connecting: host|SELECT|select| *CREATE|create|/\*{1}\*+)[\s\S]*?(?<!,)(\n\n)(?!( *[A-Z][A-Z]+)| *\-{2})'
        # (\s*(private|public|protected) ((void|static) ){0,1}[A-Z][a-zA-Z]+.*(\)|\}|;|\(|\s*\{)(?=\n)\n)+
        version_regex = '[A-Z]+[A-Za-z]{4,10} (\d+\.)+\d*'
        # | ( *([ ^\s]+ +){0, 1}[^ \s]+? {0, 4}\= {0, 4}.*)
        # (for )|(if *\(.*?\))|(//.*)
        refine_sentence = sentence

        if entity:
            formula_pattern = re.compile(formula_regex)
            url_pattern = re.compile(url_regex)
            js_pattern = re.compile(js_regex)
            css_pattern = re.compile(css_regex)
            php_pattern = re.compile(php_regex)
            html_pattern = re.compile(html_regex)
            html1_pattern = re.compile(html1_regex)
            html2_pattern = re.compile(html2_regex)
            html3_pattern = re.compile(html3_regex)
            latex_pattern = re.compile(latex_regex)
            formula1_pattern = re.compile(formula1_regex)
            latex1_pattern = re.compile(latex1_regex)
            latex2_pattern = re.compile(latex2_regex)
            java_pattern = re.compile(java_regex)
            java2_pattern = re.compile(java2_regex)
            java3_pattern = re.compile(java3_regex)
            java4_pattern = re.compile(java4_regex)
            java5_pattern = re.compile(java5_regex)
            c_pattern = re.compile(c_regex)
            c2_pattern = re.compile(c2_regex)
            python_pattern = re.compile(python_regex)
            sql_pattern = re.compile(sql_regex)
            # print(refine_sentence)
            refine_sentence = re.sub(pattern=formula_pattern, repl='<formu>', string=refine_sentence)
            refine_sentence = re.sub(pattern=url_pattern, repl='<url>', string=refine_sentence)
            refine_sentence = re.sub(pattern=css_pattern, repl='<css>', string=refine_sentence)
            refine_sentence = re.sub(pattern=js_pattern, repl='<js>', string=refine_sentence)
            refine_sentence = re.sub(pattern=php_pattern, repl='<php>', string=refine_sentence)
            refine_sentence = re.sub(pattern=html_pattern, repl='<html>', string=refine_sentence)
            refine_sentence = re.sub(pattern=html1_pattern, repl='<html>', string=refine_sentence)
            refine_sentence = re.sub(pattern=html2_pattern, repl='<html>', string=refine_sentence)
            refine_sentence = re.sub(pattern=html3_pattern, repl='<html>', string=refine_sentence)
            refine_sentence = re.sub(pattern=latex_pattern, repl='<latex>', string=refine_sentence)
            refine_sentence = re.sub(pattern=formula1_pattern, repl='<formu>', string=refine_sentence)
            refine_sentence = re.sub(pattern=latex1_pattern, repl='<latex>', string=refine_sentence)
            refine_sentence = re.sub(pattern=latex2_pattern, repl='<latex>', string=refine_sentence)
            refine_sentence = re.sub(pattern=java_pattern, repl='<java>', string=refine_sentence)
            refine_sentence = re.sub(pattern=c_pattern, repl='<c>', string=refine_sentence)
            refine_sentence = re.sub(pattern=c2_pattern, repl='<c>', string=refine_sentence)
            # print(refine_sentence)
            refine_sentence = re.sub(pattern=java2_pattern, repl='<java>', string=refine_sentence)
            refine_sentence = re.sub(pattern=java3_pattern, repl='<java>', string=refine_sentence)
            refine_sentence = re.sub(pattern=java4_pattern, repl='<java>', string=refine_sentence)
            refine_sentence = re.sub(pattern=java5_pattern, repl='<java>', string=refine_sentence)
            refine_sentence = re.sub(pattern=python_pattern, repl='<python>', string=refine_sentence)
            refine_sentence = re.sub(pattern=sql_pattern, repl='<sql>', string=refine_sentence)

        print(refine_sentence)
        print('---------------------------------------------')
        # print(re.sub(pattern='', repl='', string=refine_sentence))

        if punc:
            punc_regex = '(' + punc_regex + ')'

        pattern = re.compile(punc_regex)

        if entity:
            refine_sentence = ' '.join(
                re.split(pattern=re.compile('(<.*?>)'), string=refine_sentence))  # split entity

        refine_sentence = ' '.join(re.split(pattern=re.compile('([A-z]\.\.+([A-z]|\(|\[|\{))'), string=refine_sentence))  # split ...
        refine_sentence = ' ' + refine_sentence.strip().replace('\n', ' ').replace(' ', '  ') + ' '
        phrase = list(re.split(pattern=pattern, string=refine_sentence))
        words = []
        for index in range(len(phrase) - 1):
            words += re.split(pattern=re.compile('\s|/'), string=phrase[index].strip())

        while '' in words:
            words.remove('')

        return words

    def trim_input(self, title, question, answer, t_max_len=30, q_max_len=239, a_max_len=239):
        max_sequence_length = self.MAX_LEN

        t = self.tokenizer.tokenize(title)
        q = self.tokenizer.tokenize(question)
        a = self.tokenizer.tokenize(answer)

        t_len = len(t)
        q_len = len(q)
        a_len = len(a)

        if (t_len + q_len + a_len + 4) > max_sequence_length:

            if t_max_len > t_len:
                t_new_len = t_len
                a_max_len = a_max_len + floor((t_max_len - t_len) / 2)
                q_max_len = q_max_len + ceil((t_max_len - t_len) / 2)
            else:
                t_new_len = t_max_len

            if a_max_len > a_len:
                a_new_len = a_len
                q_new_len = q_max_len + (a_max_len - a_len)
            elif q_max_len > q_len:
                a_new_len = a_max_len + (q_max_len - q_len)
                q_new_len = q_len
            else:
                a_new_len = a_max_len
                q_new_len = q_max_len

            if t_new_len + a_new_len + q_new_len + 4 != max_sequence_length:
                raise ValueError("New sequence length should be %d, but is %d"
                                 % (max_sequence_length, (t_new_len + a_new_len + q_new_len + 4)))

            t = t[:t_new_len]
            q = q[:q_new_len]
            a = a[:a_new_len]

        return t, q, a

    def get_token_ids(self, row):
        t_tokens, q_tokens, a_tokens = self.trim_input(row.question_title, row.question_body, row.answer)

        tokens = ['[CLS]'] + t_tokens + ['[SEP]'] + q_tokens + ['[SEP]'] + a_tokens + ['[SEP]']
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) < self.MAX_LEN:
            token_ids += [0] * (self.MAX_LEN - len(token_ids))
        ids = torch.tensor(token_ids)
        seg_ids = self.get_seg_ids(ids)
        return ids, seg_ids

    def get_seg_ids(self, ids):
        seg_ids = torch.zeros_like(ids)
        seg_idx = 0
        first_sep = True
        for i, e in enumerate(ids):
            seg_ids[i] = seg_idx
            if e == self.SEP_TOKEN_ID:
                if first_sep:
                    first_sep = False
                else:
                    seg_idx = 1
        pad_idx = torch.nonzero(ids == 0)
        seg_ids[pad_idx] = 0

        return seg_ids

    def get_label(self, row):
        return torch.tensor(row[self.target_cols].values.astype(np.float32))

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        seg_ids = torch.stack([x[1] for x in batch])

        if self.labeled:
            labels = torch.stack([x[2] for x in batch])
            return token_ids, seg_ids, labels
        else:
            return token_ids, seg_ids


# In[ ]:


def get_test_loader(batch_size=4):
    df = pd.read_csv(filepath_or_buffer='../input/google-quest-challenge/test.csv', header=0)
    ds_test = QuestDataset(df, train_mode=False, labeled=False)
    loader = data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                         collate_fn=ds_test.collate_fn, drop_last=False)
    loader.num = len(df)

    return loader


def get_train_val_loaders(batch_size=4, val_batch_size=4, ifold=0):
    df = pd.read_csv(filepath_or_buffer='../input/google-quest-challenge/train.csv', header=0)
    df = shuffle(df, random_state=RANDOM_SEED)

    df_train = None
    df_val = None

    # split_index = int(len(df) * (1-val_percent))
    gkf = GroupKFold(n_splits=SPLIT_NUM).split(X=df.question_body, groups=df.question_body)
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        # print(str(fold), str((train_idx.shape, valid_idx.shape)))
        if fold == ifold:
            df_train = df.iloc[train_idx]
            df_val = df.iloc[valid_idx]
            break

    ds_train = QuestDataset(df_train)
    train_loader = data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=ds_train.collate_fn, drop_last=True)
    train_loader.num = len(df_train)

    ds_val = QuestDataset(df_val, train_mode=False)
    val_loader = data.DataLoader(ds_val, batch_size=val_batch_size, shuffle=False, num_workers=0, collate_fn=ds_val.collate_fn, drop_last=False)
    val_loader.num = len(df_val)
    val_loader.df = df_val

    return train_loader, val_loader


def test_train_loader():
    loader, _ = get_train_val_loaders(4, 4, 1)
    for ids, seg_ids, labels in loader:
        print(ids)
        print(ids.dtype)
        print(seg_ids)
        print(seg_ids.dtype)
        print(labels.dtype)
        break


def test_test_loader():
    loader = get_test_loader(4)
    for ids, seg_ids in loader:
        print(ids.dtype)
        print(seg_ids.dtype)
        break


# In[ ]:


#test_train_loader()


# In[ ]:


from transformers import BertModel, AdamW, BertConfig, get_linear_schedule_with_warmup, BertPreTrainedModel, BertConfig
import torch.nn as nn
import random
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

bert_model_config = '../input/pretrained-bert-models-for-pytorch/bert-base-uncased/bert_config.json'
bert_config = BertConfig.from_json_file(bert_model_config)
bert_config.num_labels = 30


# In[ ]:


class QuestModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.high_dropout = nn.Dropout(p=0.5)
        
        n_weights = config.num_hidden_layers + 1
        weights_init = torch.zeros(n_weights).float()
        weights_init.data[:-1] = -3
        self.layer_weights = torch.nn.Parameter(weights_init)
        
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, ids, seg_ids, mask, target=None):
        outputs = self.bert(
            input_ids=ids,
            token_type_ids=seg_ids,
            attention_mask=mask,
        )
        
        hidden_layers = outputs[2]
        last_hidden = outputs[0]

        cls_outputs = torch.stack(
            [self.dropout(layer[:, 0, :]) for layer in hidden_layers], dim=2
        )
        cls_output = (torch.softmax(self.layer_weights, dim=0) * cls_outputs).sum(-1)

        # multisample dropout (wut): https://arxiv.org/abs/1905.09788
        logits = torch.mean(
            torch.stack(
                [self.classifier(self.high_dropout(cls_output)) for _ in range(5)],
                dim=0,
            ),
            dim=0,
        )

        outputs = logits
        
# #         pooled_output = outputs[1]
#         #hidden_state = outputs[0]
#         #pooled_output = hidden_state[:, 0]
#         #pooled_output = self.pre_classifier(pooled_output)
#         #pooled_output = nn.SELU()(pooled_output)
#         pooled_output = torch.mean(outputs[0], 1)

#         pooled_output = self.dropout(pooled_output)
#         logits = self.classifier(pooled_output)
# #         logits = torch.sigmoid(logits)

#         outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if target is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), target.view(-1, self.num_labels))
            outputs = (loss, outputs)

        return outputs  # (loss), logits, (hidden_states), (attentions)


# In[ ]:


def test_model():
    x = torch.tensor([[1, 2, 3, 4, 5, 0, 0], [1, 2, 3, 4, 5, 0, 0]])
    seg_ids = torch.tensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
    model = QuestModel.from_pretrained(
        "../input/pretrained-bert-models-for-pytorch/bert-base-uncased/",  # Use the 12-layer BERT model, with an uncased vocab.
        config=bert_config
    )
    model.cuda()
    y = model(ids=x.cuda(), seg_ids=seg_ids.cuda(), mask=(x > 0).long().cuda())

    output_dir = 'test_model_save'

    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Saving model to %s" % output_dir)

    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(output_dir)

    print(y[0])
    print(y)
    
    model.cpu()
    torch.cuda.empty_cache()
    
    del model


# In[ ]:


#test_model()


# In[ ]:


torch.cuda.empty_cache()


# In[ ]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def spearmanr_accuracy(preds, target):
    total_acc = 0
    for i in range(preds.shape[0]):
        accruacy, _ = spearmanr(preds[i], target[i], axis=0, nan_policy='propagate')
        total_acc += accruacy

    return total_acc/preds.shape[0]


def train(device):

    batch_size = 16

    for ifold in range(SPLIT_NUM):
        train_loader, val_loader = get_train_val_loaders(batch_size=batch_size, val_batch_size=batch_size, ifold=ifold)
        
        model = QuestModel.from_pretrained(
            "../input/pretrained-bert-models-for-pytorch/bert-base-uncased/",
            config=bert_config
        )
    
        model.to(device)

        params = list(model.named_parameters())

        print('The BERT model has {:} different named parameters.\n'.format(len(params)))

        print('==== Embedding Layer ====\n')

        for p in params[0:5]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== First Transformer ====\n')

        for p in params[5:21]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        print('\n==== Output Layer ====\n')

        for p in params[-4:]:
            print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

        # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
        # I believe the 'W' stands for 'Weight Decay fix"
        optimizer = AdamW(model.parameters(),
                          lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                          # Candidate 5e-5, 3e-5, 2e-5
                          eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                          )

        # Number of training epochs (authors recommend between 2 and 4)
        epochs = 8

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_loader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        # Store the average loss after each epoch so we can plot them.
        train_loss_values = []
        train_acc_values = []
        val_loss_values = []
        val_acc_values = []

        # Set the seed value all over the place to make this reproducible.
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        for epoch_i in range(0, epochs):

            # ========================================
            #               Training
            # ========================================

            # Perform one full pass over the training set.

            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            # Reset the total loss for this epoch.
            total_loss = 0
            train_preds = None
            train_targets = None

            # For each batch of training data...
            for step, batch in enumerate(train_loader):

                # Progress update every 40 batches.
                if step % 40 == 0 and not step == 0:
                    # Calculate elapsed time in minutes.
                    elapsed = format_time(time.time() - t0)

                    # Report progress.
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

                input_ids = batch[0]
                input_seg = batch[1]
                input_mask = (input_ids > 0)
                target = batch[2]

                # Always clear any previously calculated gradients before performing a
                # backward pass. PyTorch doesn't do this automatically because
                # accumulating the gradients is "convenient while training RNNs".
                # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
                # model.zero_grad()

                outputs = model(
                    ids=input_ids.to(device),
                    seg_ids=input_seg.to(device),
                    mask=input_mask.to(device),
                    target=target.to(device),
                )

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss, logits = outputs[:2]
                
                optimizer.zero_grad()

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                target_ids = target.to('cpu').numpy()

                if train_preds is None:
                    train_preds = logits
                    train_targets = target_ids
                else:
                    train_preds = np.append(train_preds, logits, axis=0)
                    train_targets = np.append(train_targets, target_ids, axis=0)

                # if step > 10:
                #     break

            # Calculate the average loss over the training data.
            avg_train_loss = total_loss / len(train_loader)

            # Store the loss value for plotting the learning curve.
            train_loss_values.append(avg_train_loss)

            # Report the final accuracy for this validation run.
            train_accuracy = spearmanr_accuracy(train_preds, train_targets)
            train_acc_values.append(train_accuracy)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Average training acc: {0:.2f}".format(train_accuracy))
            print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))

            # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.

            print("")
            print("Running Validation...")

            t0 = time.time()

            # Put the model in evaluation mode--the dropout layers behave differently
            # during evaluation.
            model.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            val_preds = None
            val_targets = None

            eval_total_loss = 0

            # Evaluate data for one epoch
            for batch in val_loader:
                # Unpack the inputs from our dataloader
                input_ids = batch[0]
                input_seg = batch[1]
                input_mask = (input_ids > 0)
                target = batch[2]

                # Telling the model not to compute or store gradients, saving memory and
                # speeding up validation
                with torch.no_grad():
                    outputs = model(
                        ids=input_ids.to(device),
                        seg_ids=input_seg.to(device),
                        mask=input_mask.to(device),
                        target=target.to(device),
                    )

                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                loss, logits = outputs[:2]

                eval_total_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                target_ids = target.to('cpu').numpy()

                nb_eval_steps += 1

                if val_preds is None:
                    val_preds = logits
                    val_targets = target_ids
                else:
                    val_preds = np.append(val_preds, logits, axis=0)
                    val_targets = np.append(val_targets, target_ids, axis=0)

                # if nb_eval_steps > 10:
                #     break

            # Calculate the average loss over the training data.
            avg_val_loss = total_loss / len(val_loader)

            # Store the loss value for plotting the learning curve.
            val_loss_values.append(avg_val_loss)

            # Report the final accuracy for this validation run.
            eval_accuracy = spearmanr_accuracy(val_preds, val_targets)
            val_acc_values.append(eval_accuracy)

            print("Validation Loss: {0:.2f}".format(avg_val_loss))
            print("Validation Accuracy: {0:.2f}".format(eval_accuracy))
            print("Validation took: {:}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete!")

        output_dir = 'model_fold{0}.pt'.format(ifold+1)

        torch.save(model.state_dict(), output_dir)
        model.cpu()
        torch.cuda.empty_cache()


def predict(device):
    batch_size = 16
    test_loader = get_test_loader(batch_size=batch_size)

    prediction = None

    for ifold in range(SPLIT_NUM):
        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(test_loader.num))
        model = QuestModel(bert_config)
        model.load_state_dict(torch.load('../input/custombert/model_fold{0}.pt'.format(ifold+1)))

        # Put model in evaluation mode
        model.eval()
        model.to(device)

        # Tracking variables
        preds = None

        # Predict
        for batch in test_loader:
            # Unpack the inputs from our dataloader
            input_ids = batch[0]
            input_seg = batch[1]
            input_mask = (input_ids > 0)
            # target = batch[2].float().detach().cpu().numpy()

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                outputs = model(
                    ids=input_ids.to(device),
                    seg_ids=input_seg.to(device),
                    mask=input_mask.to(device),
                )

            logits = outputs
            logits = logits.sigmoid()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Store predictions and true labels
            if preds is None:
                preds = logits
            else:
                preds = np.append(preds, logits, axis=0)

        if prediction is None:
            prediction = preds
        else:
            prediction += preds

        print('Goup {0}\tDONE.'.format(ifold))

    return prediction/SPLIT_NUM


# In[ ]:


print('The torch version is ' + str(torch.__version__))

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


# In[ ]:


# train(device)


# In[ ]:


preds = predict(device)

test_df = pd.read_csv('../input/google-quest-challenge/test.csv')
submission_df = pd.read_csv('../input/google-quest-challenge/sample_submission.csv')

preds_df = pd.DataFrame(data=preds, columns=submission_df.columns[-30:])
submission_df = pd.concat([test_df['qa_id'], preds_df], axis=1)

submission_df.to_csv('submission.csv', index=False)

