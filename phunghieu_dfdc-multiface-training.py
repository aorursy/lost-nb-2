#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.nn import functional as F
from torchvision.models import resnet18
from albumentations import Normalize, Compose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import os
import glob
import multiprocessing as mp

if torch.cuda.is_available():
    device = 'cuda:0'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
print(f'Running on device: {device}')


# In[2]:


INPUT_DIR = '/kaggle/input/'
SAVE_PATH = '/kaggle/working/f5_resnet18.pth' # The location where the model should be saved.
PRETRAINED_MODEL_PATH = ''

N_FACES = 5
TEST_SIZE = 0.3
RANDOM_STATE = 123

BATCH_SIZE = 32
NUM_WORKERS = mp.cpu_count()

WARM_UP_EPOCHS = 10
WARM_UP_LR = 1e-4
FINE_TUNE_EPOCHS = 100
FINE_TUNE_LR = 1e-6

THRESHOLD = 0.5
EPSILON = 1e-7


# In[3]:


def calculate_f1(preds, labels):
    '''
    Parameters:
        preds: The predictions.
        labels: The labels.

    Returns:
        f1 score
    '''

    labels = np.array(labels, dtype=np.uint8)
    preds = (np.array(preds) >= THRESHOLD).astype(np.uint8)
    tp = np.count_nonzero(np.logical_and(labels, preds))
    tn = np.count_nonzero(np.logical_not(np.logical_or(labels, preds)))
    fp = np.count_nonzero(np.logical_not(labels)) - tn
    fn = np.count_nonzero(labels) - tp
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = (2 * precision * recall) / (precision + recall + EPSILON)
    
    return f1


def train_the_model(
    model,
    criterion,
    optimizer,
    epochs,
    train_dataloader,
    val_dataloader,
    best_val_loss=1e7,
    best_val_logloss=1e7,
    save_the_best_on='val_logloss'
):
    '''
    Parameters:
        model: The model needs to be trained.
        criterion: Loss function.
        optimizer: The optimizer.
        epochs: The number of epochs
        train_dataloader: The dataloader used to generate training samples.
        val_dataloader: The dataloader used to generate validation samples.
        best_val_loss: The initial value of the best val loss (default: 1e7.)
        best_val_logloss: The initial value of the best val log loss (default: 1e7.)
        save_the_best_on: Whether to save the best model based on "val_loss" or "val_logloss" (default: val_logloss.)

    Returns:
        losses: All computed losses.
        val_losses: All computed val_losses.
        loglosses: All computed loglosses.
        val_loglosses: All computed val_loglosses.
        f1_scores: All computed f1_scores.
        val_f1_scores: All computed val_f1_scores.
        best_val_loss: New value of the best val loss.
        best_val_logloss: New value of the best val log loss.
        best_model_state_dict: The state_dict of the best model.
        best_optimizer_state_dict: The state_dict of the optimizer corresponds to the best model.
    '''

    losses = np.zeros(epochs)
    val_losses = np.zeros(epochs)
    loglosses = np.zeros(epochs)
    val_loglosses = np.zeros(epochs)
    f1_scores = np.zeros(epochs)
    val_f1_scores = np.zeros(epochs)
    best_model_state_dict = None
    best_optimizer_state_dict = None

    logloss = nn.BCELoss()

    for i in tqdm(range(epochs)):
        batch_losses = []
        train_pbar = tqdm(train_dataloader)
        train_pbar.desc = f'Epoch {i+1}'
        classifier.train()

        all_labels = []
        all_preds = []

        for i_batch, sample_batched in enumerate(train_pbar):
            # Make prediction.
            y_pred = classifier(sample_batched['faces'])

            all_labels.extend(sample_batched['label'].squeeze(dim=-1).tolist())
            all_preds.extend(y_pred.squeeze(dim=-1).tolist())

            # Compute loss.
            loss = criterion(y_pred, sample_batched['label'])
            batch_losses.append(loss.item())

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Display some information in progress-bar.
            train_pbar.set_postfix({
                'loss': batch_losses[-1]
            })

        # Compute scores.
        loglosses[i] = logloss(torch.tensor(all_preds).to(device), torch.tensor(all_labels).to(device))
        f1_scores[i] = calculate_f1(all_preds, all_labels)

        # Compute batch loss (average).
        losses[i] = np.array(batch_losses).mean()


        # Compute val loss
        val_batch_losses = []
        val_pbar = tqdm(val_dataloader)
        val_pbar.desc = 'Validating'
        classifier.eval()

        all_labels = []
        all_preds = []

        for i_batch, sample_batched in enumerate(val_pbar):
            # Make prediction.
            y_pred = classifier(sample_batched['faces'])

            all_labels.extend(sample_batched['label'].squeeze(dim=-1).tolist())
            all_preds.extend(y_pred.squeeze(dim=-1).tolist())

            # Compute val loss.
            val_loss = criterion(y_pred, sample_batched['label'])
            val_batch_losses.append(val_loss.item())

            # Display some information in progress-bar.
            val_pbar.set_postfix({
                'val_loss': val_batch_losses[-1]
            })

        # Compute val scores.
        val_loglosses[i] = logloss(torch.tensor(all_preds).to(device), torch.tensor(all_labels).to(device))
        val_f1_scores[i] = calculate_f1(all_preds, all_labels)

        val_losses[i] = np.array(val_batch_losses).mean()
        print(f'loss: {losses[i]} | val loss: {val_losses[i]} | f1: {f1_scores[i]} | val f1: {val_f1_scores[i]} | log loss: {loglosses[i]} | val log loss: {val_loglosses[i]}')
        
        # Update the best values
        if val_losses[i] < best_val_loss:
            best_val_loss = val_losses[i]
            if save_the_best_on == 'val_loss':
                print('Found a better checkpoint!')
                best_model_state_dict = classifier.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
        if val_loglosses[i] < best_val_logloss:
            best_val_logloss = val_loglosses[i]
            if save_the_best_on == 'val_logloss':
                print('Found a better checkpoint!')
                best_model_state_dict = classifier.state_dict()
                best_optimizer_state_dict = optimizer.state_dict()
            
    return losses, val_losses, loglosses, val_loglosses, f1_scores, val_f1_scores, best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict


def visualize_results(
    losses,
    val_losses,
    loglosses,
    val_loglosses,
    f1_scores,
    val_f1_scores
):
    '''
    Parameters:
        losses: A list of losses.
        val_losses: A list of val losses.
        loglosses: A list of loglosses.
        val_loglosses: A list of val loglosses.
        f1_scores: A list of f1 scores.
        val_f1_scores: A list of val f1 scores.
    '''

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(losses) + 1), losses)
    ax.plot(np.arange(1, len(val_losses) + 1), val_losses)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('focal loss', fontsize='xx-large')
    ax.legend(
        ['loss', 'val loss'],
        loc='upper right',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()

    
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(loglosses) + 1), loglosses)
    ax.plot(np.arange(1, len(val_loglosses) + 1), val_loglosses)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('log loss', fontsize='xx-large')
    ax.legend(
        ['log loss', 'val log loss'],
        loc='upper right',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()


    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_axes([0, 0, 1, 1])

    ax.plot(np.arange(1, len(f1_scores) + 1), f1_scores)
    ax.plot(np.arange(1, len(val_f1_scores) + 1), val_f1_scores)
    ax.set_xlabel('epoch', fontsize='xx-large')
    ax.set_ylabel('f1 score', fontsize='xx-large')
    ax.legend(
        ['f1', 'val f1'],
        loc='upper left',
        fontsize='xx-large',
        shadow=True
    )
    plt.show()


# In[4]:


class DeepfakeClassifier(nn.Module):
    def __init__(self, encoder, in_channels=3, num_classes=1):
        super(DeepfakeClassifier, self).__init__()
        self.encoder = encoder
        
        # Modify input layer.
        self.encoder.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        
        # Modify output layer.
        self.encoder.fc = nn.Linear(512 * 1, num_classes)

    def forward(self, x):
        return torch.sigmoid(self.encoder(x))
    
    def freeze_all_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def freeze_middle_layers(self):
        self.freeze_all_layers()
        
        for param in self.encoder.conv1.parameters():
            param.requires_grad = True
            
        for param in self.encoder.fc.parameters():
            param.requires_grad = True

    def unfreeze_all_layers(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


class FaceDataset(Dataset):
    def __init__(self, img_dirs, labels, n_faces=1, preprocess=None):
        self.img_dirs = img_dirs
        self.labels = labels
        self.n_faces = n_faces
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        face_paths = glob.glob(f'{img_dir}/*.png')

        if len(face_paths) >= self.n_faces:
            sample = np.random.choice(face_paths, self.n_faces, replace=False)
        else:
            sample = np.random.choice(face_paths, self.n_faces, replace=True)
            
        faces = []
        
        for face_path in sample:
            face = cv2.imread(face_path, 1)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            if self.preprocess is not None:
                augmented = self.preprocess(image=face)
                face = augmented['image']
            faces.append(face)

        return {'faces': np.concatenate(faces, axis=-1).transpose(2, 0, 1), 'label': np.array([label], dtype=float)}
    
    
class FaceValDataset(Dataset):
    def __init__(self, img_dirs, labels, n_faces=1, preprocess=None):
        self.img_dirs = img_dirs
        self.labels = labels
        self.n_faces = n_faces
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.img_dirs[idx]
        label = self.labels[idx]
        face_paths = glob.glob(f'{img_dir}/*.png')

        face_indices = [
            path.split('/')[-1].split('.')[0].split('_')[0]
            for path in face_paths
        ]        
        max_idx = np.max(np.array(face_indices, dtype=np.uint32))

        selected_paths = []

        for i in range(self.n_faces):
            stride = int((max_idx + 1)/(self.n_faces**2))
            sample = np.linspace(i*stride, max_idx + i*stride, self.n_faces).astype(int)

            # Get faces
            for idx in sample:
                paths = glob.glob(f'{img_dir}/{idx}*.png')

                selected_paths.extend(paths)

                if len(selected_paths) >= self.n_faces:
                    break
            
            if len(selected_paths) >= self.n_faces:
                break

        faces = []

        selected_paths = selected_paths[:self.n_faces] # Get top
        for selected_path in selected_paths:
            img = cv2.imread(selected_path, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces.append(img)

        if self.preprocess is not None:
            for j in range(len(faces)):
                augmented = self.preprocess(image=faces[j])
                faces[j] = augmented['image']

        faces = np.concatenate(faces, axis=-1).transpose(2, 0, 1)

        return {
            'faces': faces,
            'label': np.array([label], dtype=float)
        }


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, sample_weight=None):
        super().__init__()
        self.gamma = gamma
        self.sample_weight = sample_weight

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        if self.sample_weight is not None:
            loss = loss * self.sample_weight
        return loss.mean()


# In[5]:


all_train_dirs = glob.glob(INPUT_DIR + 'deepfake-detection-faces-*')
all_train_dirs = sorted(all_train_dirs, key=lambda x: x)
for i, train_dir in enumerate(all_train_dirs):
    print('[{:02}]'.format(i), train_dir)


# In[6]:


all_dataframes = []
for train_dir in all_train_dirs:
    df = pd.read_csv(os.path.join(train_dir, 'metadata.csv'))
    df['path'] = df['filename'].apply(lambda x: os.path.join(train_dir, x.split('.')[0]))
    all_dataframes.append(df)

train_df = pd.concat(all_dataframes, ignore_index=True, sort=False)


# In[7]:


train_df.head()


# In[8]:


# Remove empty folders
train_df = train_df[train_df['path'].map(lambda x: os.path.exists(x))]


# In[9]:


train_df.head()


# In[10]:


valid_train_df = pd.DataFrame(columns=['filename', 'label', 'split', 'original', 'path'])

# for row_idx, row in tqdm(train_df.iterrows()):
for row_idx in tqdm(train_df.index):
    row = train_df.loc[row_idx]
    img_dir = row['path']
    face_paths = glob.glob(f'{img_dir}/*.png')

    if len(face_paths) >= N_FACES: # Satisfy the minimum requirement for the number of faces
        face_indices = [
            path.split('/')[-1].split('.')[0].split('_')[0]
            for path in face_paths
        ]
        max_idx = np.max(np.array(face_indices, dtype=np.uint32))

        selected_paths = []

        for i in range(N_FACES):
            stride = int((max_idx + 1)/(N_FACES**2))
            sample = np.linspace(i*stride, max_idx + i*stride, N_FACES).astype(int)

            # Get faces
            for idx in sample:
                paths = glob.glob(f'{img_dir}/{idx}*.png')

                selected_paths.extend(paths)
                if len(selected_paths) >= N_FACES: # Get enough faces
                    break

            if len(selected_paths) >= N_FACES: # Get enough faces
                valid_train_df = valid_train_df.append(row, ignore_index=True)
                break


# In[11]:


valid_train_df.head()


# In[12]:


valid_train_df['label'].replace({'FAKE': 1, 'REAL': 0}, inplace=True)


# In[13]:


valid_train_df.head()


# In[14]:


label_count = valid_train_df.groupby('label').count()['filename']
print(label_count)


# In[15]:


X = valid_train_df['path'].to_numpy()
y = valid_train_df['label'].to_numpy()


# In[16]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


# In[17]:


preprocess = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1)
])


# In[18]:


train_dataset = FaceDataset(
    img_dirs=X_train,
    labels=y_train,
    n_faces=N_FACES,
    preprocess=preprocess
)
val_dataset = FaceValDataset(
    img_dirs=X_val,
    labels=y_val,
    n_faces=N_FACES,
    preprocess=preprocess
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# In[19]:


if os.path.exists(PRETRAINED_MODEL_PATH):
    encoder = resnet18(pretrained=False)
    classifier = DeepfakeClassifier(encoder=encoder, in_channels=3*N_FACES, num_classes=1)
    state = torch.load(PRETRAINED_MODEL_PATH, map_location=lambda storage, loc: storage)
    classifier.load_state_dict(state['state_dict'])
else:
    encoder = resnet18(pretrained=True)
    classifier = DeepfakeClassifier(encoder=encoder, in_channels=3*N_FACES, num_classes=1)

classifier.to(device)
classifier.train()


# In[20]:


criterion = FocalLoss()


# In[21]:


losses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_losses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
loglosses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_loglosses = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
f1_scores = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)
val_f1_scores = np.zeros(WARM_UP_EPOCHS + FINE_TUNE_EPOCHS)

if os.path.exists(PRETRAINED_MODEL_PATH):
    best_val_loss = state['best_val_loss']
else:
    best_val_loss = 1e7

if os.path.exists(PRETRAINED_MODEL_PATH):
    best_val_logloss = state['best_val_logloss']
else:
    best_val_logloss = 1e7


# In[22]:


classifier.freeze_middle_layers()


# In[23]:


warmup_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=WARM_UP_LR)
if os.path.exists(PRETRAINED_MODEL_PATH) and 'warmup_optimizer' in state.keys():
    warmup_optimizer.load_state_dict(state['warmup_optimizer'])


# In[24]:


losses[:WARM_UP_EPOCHS], val_losses[:WARM_UP_EPOCHS], loglosses[:WARM_UP_EPOCHS], val_loglosses[:WARM_UP_EPOCHS], f1_scores[:WARM_UP_EPOCHS], val_f1_scores[:WARM_UP_EPOCHS], best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict = train_the_model(
    model=classifier,
    criterion=criterion,
    optimizer=warmup_optimizer,
    epochs=WARM_UP_EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    best_val_loss=best_val_loss,
    best_val_logloss=best_val_logloss,
    save_the_best_on='val_logloss'
)

# Save the best checkpoint.
if best_model_state_dict is not None:
    state = {
        'state_dict': best_model_state_dict,
        'warmup_optimizer': best_optimizer_state_dict,
        'best_val_loss': best_val_loss,
        'best_val_logloss': best_val_logloss
    }

    torch.save(state, SAVE_PATH)


# In[25]:


visualize_results(
    losses=losses[:WARM_UP_EPOCHS],
    val_losses=val_losses[:WARM_UP_EPOCHS],
    loglosses=loglosses[:WARM_UP_EPOCHS],
    val_loglosses=val_loglosses[:WARM_UP_EPOCHS],
    f1_scores=f1_scores[:WARM_UP_EPOCHS],
    val_f1_scores=val_f1_scores[:WARM_UP_EPOCHS]
)


# In[26]:


classifier.unfreeze_all_layers()


# In[27]:


finetune_optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=FINE_TUNE_LR)
if os.path.exists(PRETRAINED_MODEL_PATH) and 'finetune_optimizer' in state.keys() and WARM_UP_EPOCHS == 0:
    finetune_optimizer.load_state_dict(state['finetune_optimizer'])


# In[28]:


losses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_losses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], loglosses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_loglosses[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], f1_scores[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], val_f1_scores[WARM_UP_EPOCHS:WARM_UP_EPOCHS+FINE_TUNE_EPOCHS], best_val_loss, best_val_logloss, best_model_state_dict, best_optimizer_state_dict = train_the_model(
    model=classifier,
    criterion=criterion,
    optimizer=finetune_optimizer,
    epochs=FINE_TUNE_EPOCHS,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    best_val_loss=best_val_loss,
    best_val_logloss=best_val_logloss,
    save_the_best_on='val_logloss'
)

# Save the best checkpoint.
if best_model_state_dict is not None:
    state = {
        'state_dict': best_model_state_dict,
        'finetune_optimizer': best_optimizer_state_dict,
        'best_val_loss': best_val_loss,
        'best_val_logloss': best_val_logloss
    }

    torch.save(state, SAVE_PATH)


# In[29]:


visualize_results(
    losses=losses,
    val_losses=val_losses,
    loglosses=loglosses,
    val_loglosses=val_loglosses,
    f1_scores=f1_scores,
    val_f1_scores=val_f1_scores
)

