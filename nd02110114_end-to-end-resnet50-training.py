#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pretrainedmodels iterative-stratification > /dev/null')


# In[2]:


import os
import gc
import cv2
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pretrainedmodels
import albumentations as albu
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from catalyst.utils import get_device, set_global_seed
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import recall_score


# In[3]:


def read_data(BASE_PATH):
    print('Reading train.csv file....')
    train = pd.read_csv(BASE_PATH + 'train.csv')
    print('Training.csv file have {} rows and {} columns'.format(
        train.shape[0], train.shape[1]))

    print('Reading test.csv file....')
    test = pd.read_csv(BASE_PATH + 'test.csv')
    print('Test.csv file have {} rows and {} columns'.format(
        test.shape[0], test.shape[1]))

    print('Reading sample_submission.csv file....')
    sample_submission = pd.read_csv(BASE_PATH + 'sample_submission.csv')
    print('Sample_submission.csv file have {} rows and {} columns'.format(
        sample_submission.shape[0], sample_submission.shape[1]))
    return train, test, sample_submission

def prepare_image(datadir, data_type='train', submission=False):
    assert data_type in ['train', 'test']
    images = []
    # only use 1/4 size because of CPU memory
    for i in tqdm([0]):
        if submission:
            image_df_list = pd.read_parquet(datadir + f'{data_type}_image_data_{i}.parquet')
        else:
            image_df_list = pd.read_feather(datadir + f'{data_type}_image_data_{i}.feather')

        HEIGHT = 137
        WIDTH = 236
        image = image_df_list.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)
        images.append(image)
        del image_df_list, image
        gc.collect()

    images = np.concatenate(images, axis=0)
    print('Image shape : ', images.shape)
    return images

def macro_recall(outputs, targets):
    pred_labels = [np.argmax(out, axis=1) for out in outputs]
    # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
    recall_grapheme = recall_score(targets[:, 0], pred_labels[0], average='macro')
    recall_consonant = recall_score(targets[:, 1], pred_labels[1], average='macro')
    recall_vowel = recall_score(targets[:, 2], pred_labels[2], average='macro')
    scores = [recall_grapheme, recall_consonant, recall_vowel]
    final_score = np.average(scores, weights=[2, 1, 1])
    return final_score, scores


# In[4]:


HEIGHT = 137
WIDTH = 236
SIZE = 224


def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


# https://www.kaggle.com/iafoss/image-preprocessing-128x128
def crop_resize(img, size=SIZE, pad=16):
    # crop a box around pixels large than the threshold
    # some images contain line at the sides
    ymin, ymax, xmin, xmax = bbox(img[5:-5, 5:-5] > 80)
    # cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img[ymin:ymax, xmin:xmax]
    # remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin, ymax-ymin
    l = max(lx, ly) + pad  # noqa
    # make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img, (size, size))


class BengaliAIDataset(Dataset):
    def __init__(self, images=None, labels=None, size=None, transforms=None):
        self.images = images
        self.labels = labels
        self.size = size
        self.transforms = transforms

        # set dummy labels
        if self.labels is None:
            self.labels = np.zeros(len(images))

        # validation
        if len(images) != len(labels):
            raise ValueError('Do not match the data size between input and output')

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.size is not None:
            img = crop_resize(img, self.size)
        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented['image']
        # 2dim to 3dim (N,N) -> (1,N,N)
        img = img[None, :, :]
        return torch.tensor(img, dtype=torch.float), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)
    
class LinearBlock(nn.Module):
    def __init__(self, in_features=None, out_features=None, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False):
        super(LinearBlock, self).__init__()
        # validation
        if in_features is None or out_features is None:
            raise ValueError('You should set both in_features and out_features!!')
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = self._residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h

    def _residual_add(lhs, rhs):
        lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
        if lhs_ch < rhs_ch:
            out = lhs + rhs[:, :lhs_ch]
        elif lhs_ch > rhs_ch:
            out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
        else:
            out = lhs + rhs
        return out


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            (x.size(-2), x.size(-1))).pow(1./self.p)


class BengaliBaselineClassifier(nn.Module):
    def __init__(self, n_grapheme=168, n_vowel=11, n_consonant=7,
                 pretrainedmodels=None, in_channels=1,
                 hdim=512, use_bn=True, pretrained=None):
        super(BengaliBaselineClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.base_model = pretrainedmodels
        self.conv0 = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
        inch = self.base_model.last_linear.in_features
        self.gem_pool = GeM()
        self.fc1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=F.relu)
        self.logits_for_grapheme = LinearBlock(hdim, n_grapheme, use_bn=False, activation=None)
        self.logits_for_vowel = LinearBlock(hdim, n_vowel, use_bn=False, activation=None)
        self.logits_for_consonant = LinearBlock(hdim, n_consonant, use_bn=False, activation=None)

    def forward(self, x):
        h = self.conv0(x)
        h = self.base_model.features(h)
        h = self.gem_pool(h)
        h = h.view(h.size(0), -1)
        h = self.fc1(h)
        logits_for_grapheme = self.logits_for_grapheme(h)
        logits_for_consonant = self.logits_for_consonant(h)
        logits_for_vowel = self.logits_for_vowel(h)
        # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
        logits = (logits_for_grapheme, logits_for_consonant, logits_for_vowel)
        return logits

    
class BaselineLoss(nn.Module):
    def __init__(self):
        super(BaselineLoss, self).__init__()

    def forward(self, pred, target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred[0], target[:, 0]) +             criterion(pred[1], target[:, 1]) + criterion(pred[2], target[:, 2])
        return loss


# In[5]:


class BengaliRunner:
    def __init__(self, device='cpu'):
        self.device = device

    def train(self, model, criterion, optimizer, loaders, scheduler=None, logdir=None,
              num_epochs=20, score_func=None):
        # validation
        for dict_val in [loaders]:
            if 'train' in dict_val and 'valid' in dict_val:
                pass
            else:
                raise ValueError('You should set train and valid key.')

        # setup training
        model = model.to(self.device)
        train_loader = loaders['train']
        valid_loader = loaders['valid']
        train_criterion = criterion
        valid_criterion = criterion
        best_score = -1.0
        best_avg_val_loss = 100
        log_df = pd.DataFrame(
            [], columns=['epoch', 'loss', 'valid_loss', 'score', 'recall_grapheme',
                         'recall_consonant', 'recall_vowel', 'time'],
            index=range(num_epochs)
        )
        for epoch in range(num_epochs):
            start_time = time.time()
            # release memory
            torch.cuda.empty_cache()
            gc.collect()
            # train for one epoch
            avg_loss = self._train_model(model, train_criterion, optimizer, train_loader, scheduler)
            # evaluate on validation set
            avg_val_loss, score, scores = self._validate_model(model, valid_criterion, valid_loader, score_func)

            # log
            elapsed_time = time.time() - start_time
            log_df.iloc[epoch] = [epoch + 1, avg_loss, avg_val_loss, score, scores[0], scores[1], scores[2], elapsed_time]

            # the position of this depends on the scheduler you use
            if scheduler is not None:
                scheduler.step()

            # save best params
            save_path = 'best_model.pth'
            if logdir is not None:
                save_path = os.path.join(logdir, save_path)

            if score is None:
                if best_avg_val_loss > avg_val_loss:
                    best_avg_val_loss = avg_val_loss
                    best_param_loss = model.state_dict()
                    torch.save(best_param_loss, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))
            else:
                if best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    torch.save(best_param_score, save_path)
                    print('Save the best model on Epoch {}'.format(epoch + 1))

            # save log
            log_df.to_csv(os.path.join(logdir, 'log.csv'))

        return True

    def predict_loader(self, model, loader, resume='best_model.pth'):
        # set up models
        model = model.to(self.device)
        model.load_state_dict(torch.load(resume))
        model.eval()

        # prediction
        grapheme_preds = []
        consonant_preds = []
        vowel_preds = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(images)
                logits = [out.detach().cpu().numpy() for out in output_valid]
                # target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
                grapheme_preds.extend(logits[0])
                consonant_preds.extend(logits[1])
                vowel_preds.extend(logits[2])

        return grapheme_preds, consonant_preds, vowel_preds

    def _train_model(self, model, criterion, optimizer, train_loader, scheduler=None):
        # switch to train mode
        model.train()
        avg_loss = 0.0
        for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # training
            output_train = model(images)
            loss = criterion(output_train, labels)

            # update params
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc loss
            avg_loss += loss.item() / len(train_loader)

        return avg_loss

    def _validate_model(self, model, criterion, valid_loader, score_func=None):
        # switch to eval mode
        model.eval()
        avg_val_loss = 0.
        valid_grapheme = []
        valid_consonant = []
        valid_vowel = []
        targets = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # output
                output_valid = model(images)
                avg_val_loss += criterion(output_valid, labels).item() / len(valid_loader)

                # calc score
                logits = [out.detach().cpu().numpy() for out in output_valid]
                labels = labels.detach().cpu().numpy()
                valid_grapheme.extend(logits[0])
                valid_consonant.extend(logits[1])
                valid_vowel.extend(logits[2])
                targets.extend(labels)

            score = None
            if score_func is not None:
                # TODO : you should write valid score calculation
                # In this case, we pass sigmoid function
                targets = np.array(targets)
                valid_preds = [valid_grapheme, valid_consonant, valid_vowel]
                score, scores = score_func(valid_preds, targets)

        return avg_val_loss, score, scores


# In[6]:


# set your params
BASE_PATH = '../input/bengaliai-cv19/'
DATA_PATH = '../input/bengaliaicv19feather/'
BASE_LOGDIR = './logs'
NUM_FOLDS = 5
BATCH_SIZE = 64
EPOCHS = 10
SEED = 1234
SIZE = 96
LR = 0.001
HOLD_OUT = True

# fix seed
set_global_seed(SEED)

# read dataset
train, _, _ = read_data(BASE_PATH)
train_all_images = prepare_image(DATA_PATH, data_type='train', submission=False)
train = train.iloc[0:len(train_all_images)]

# init
target_col = ['grapheme_root', 'consonant_diacritic', 'vowel_diacritic']
device = get_device()
train_data_transforms = albu.Compose([
    albu.ShiftScaleRotate(rotate_limit=10, scale_limit=.1),
    albu.Cutout(p=0.5),
])
test_data_transforms = None

# cross validation
kf = MultilabelStratifiedKFold(n_splits=NUM_FOLDS, random_state=SEED)
ids = kf.split(X=train_all_images, y=train[target_col].values)
for fold, (train_idx, valid_idx) in enumerate(ids):
    print("Current Fold: ", fold + 1)
    logdir = os.path.join(BASE_LOGDIR, 'fold_{}'.format(fold + 1))
    os.makedirs(logdir, exist_ok=True)

    train_df, valid_df = train.iloc[train_idx], train.iloc[valid_idx]
    print("Train and Valid Shapes are", train_df.shape, valid_df.shape)

    print("Preparing train datasets....")
    train_dataset = BengaliAIDataset(
        images=train_all_images[train_idx], labels=train_df[target_col].values,
        size=SIZE, transforms=train_data_transforms
    )

    print("Preparing valid datasets....")
    valid_dataset = BengaliAIDataset(
        images=train_all_images[valid_idx], labels=valid_df[target_col].values,
        size=SIZE, transforms=test_data_transforms
    )

    print("Preparing dataloaders datasets....")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loaders = {'train': train_loader, 'valid': valid_loader}

    # release memory
    del train_df, valid_df, train_dataset, valid_dataset
    gc.collect()
    torch.cuda.empty_cache()

    # init models
    resnet34 = pretrainedmodels.__dict__["resnet34"](pretrained="imagenet")
    model = BengaliBaselineClassifier(pretrainedmodels=resnet34, hdim=512)
    model = model.to(device)
    criterion = BaselineLoss()
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = None

    # catalyst trainer
    runner = BengaliRunner(device=device)
    # model training
    runner.train(model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                     loaders=loaders, logdir=logdir, num_epochs=EPOCHS, score_func=macro_recall)

    # release memory
    del model, runner, train_loader, valid_loader, loaders
    gc.collect()
    torch.cuda.empty_cache()

    if HOLD_OUT is True:
        break

