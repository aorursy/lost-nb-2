#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ---------------------------------------------------------------------------
# Import des librairies
# ---------------------------------------------------------------------------

# librairies communes
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Librairies tensorflow
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Flatten, Dense, Dropout
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# In[2]:


# ---------------------------------------------------------------------------
# Configuration des paramètres principaux du modèle
# ---------------------------------------------------------------------------
# Le chemin vers la sauvegarde du réseau
model_savepath    = 'cnn_vgg16_model_trained_2.h5'     

# Les chemins vers les jeu de données
TRAINING_DIR      = '../input/dogs-vs-cats-redux-kernels-edition/train'
TESTING_DIR       = '../input/dogs-vs-cats-redux-kernels-edition/test'

IMGSIZE       = 224    # Taille de l'image en input
EPOCH         = 22     # nombre d'epoch 
BATCH_SIZE    = 16     # traitement par batch d'images avant la descente de gradient
FREEZE_LAYERS = 15     # pour un VGG16 freeze de réapprentissage de certaines couches
TRAIN         = True   # Entrainement ou utilisation d'un réseau déjà entrainé


# In[3]:


# ---------------------------------------------------------------------------
#  Constitution des jeux de données
# ---------------------------------------------------------------------------    

# -------
#  Jeu d'entrainement
# -------

# Dataframe de deux colonnes contenant les id des fichiers et leur label
train_files = os.listdir(TRAINING_DIR)
train_labels = []

for file in train_files:
    train_labels.append(file.split(".")[0])

df_train = pd.DataFrame({"id": train_files, "label": train_labels})


# In[4]:


df_train.head()


# In[5]:


# Image generator: attention il est préférable de ne pas utiliser d'augmentation de données
# Nous utilisons également un processing spécifique au VGG16 et non pas un rescale 1./255
train_datagen =          ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.20)

train_generator =         train_datagen.flow_from_dataframe(
            df_train,
            TRAINING_DIR,
            x_col='id',
            y_col='label',
            has_ext=True,
            shuffle=True,
            target_size=(IMGSIZE, IMGSIZE),
            batch_size=BATCH_SIZE,
            subset='training',
            class_mode='categorical')

train_labels = to_categorical(train_generator.classes)


# In[6]:


# -------
#  Jeu de validation
# -------

valid_generator =         train_datagen.flow_from_dataframe(
            df_train,
            TRAINING_DIR,
            x_col='id',
            y_col='label',
            has_ext=True,
            shuffle=True,
            target_size=(IMGSIZE, IMGSIZE),
            batch_size=BATCH_SIZE,
            subset='validation',
            class_mode='categorical')


# In[7]:


# -------
# Jeu de test
# -------

test_files = os.listdir(TESTING_DIR)
df_test = pd.DataFrame({"id": test_files, 'label': 'nan'})

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)    
test_generator =     test_datagen.flow_from_dataframe(
        df_test, 
        TESTING_DIR, 
        x_col='id',
        y_col=None, 
        has_ext=True, 
        target_size=(IMGSIZE, IMGSIZE), 
        class_mode=None, 
        seed=42,
        batch_size=1, 
        shuffle=False
    )


# In[8]:


# -----------
# VGG16 pre-entrainé sans le classifier final
# https://github.com/keras-team/keras/issues/4465
# -----------

# Déclaration du modèle VGG16 (sans le top qui est le classifier)
base_model = VGG16(include_top=False, weights=None, input_tensor=None,
            input_shape=(IMGSIZE, IMGSIZE, 3))

# Chargement des points préentrainés du modèle
# Dataset = Full Keras Pretrained No Top
base_model.load_weights('../input/full-keras-pretrained-no-top/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


# In[9]:


# Classifier
x = base_model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu', name='top-fc1')(x)
x = Dense(128, activation='relu', name='top-fc2')(x)
x = Dropout(0.3)(x)

# output layer: nombre de neurones de sortie = nombre de classe a prédire
output_layer = Dense(2, activation='softmax', name='softmax')(x)


# In[10]:


# Assemblage du modèle final
net_final = Model(inputs=base_model.input, outputs=output_layer)

# freeze de certains layers (spécifique au modèle utilisé)
for layer in net_final.layers[:FREEZE_LAYERS]:
    layer.trainable = False
    
# Entrainement des derniers layers de classification
for layer in net_final.layers[FREEZE_LAYERS:]:
    layer.trainable = True

# compilation du modele
net_final.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
print(net_final.summary())


# In[11]:


# Génération des STEPS_SIZE (comme nous utilisons des générateurs infinis)
# Ceci est nécessaire pour déterminer à quel moment nous avons parcouru entiérement nos jeu de données
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST  = test_generator.n  // test_generator.batch_size


# In[12]:


if (TRAIN):
    
    # Création des Callbacks à appeler aprés chaque epoch
    #   pour sauvegarde des résultats
    checkpoint = ModelCheckpoint("model_1.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    #   pour arrêt prématuré
    early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
    
    # Entrainement du modèle
    history = net_final.fit_generator(
                    generator=train_generator,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = valid_generator,
                    validation_steps = STEP_SIZE_VALID,
                    callbacks = [checkpoint, early],
                    epochs = EPOCH)
    
    # Sauvegarde du réseau après entrainement
    net_final.save(model_savepath)    
    
else:
    net_final.load_weights('../input/cnn-vgg16-model-trained-2/cnn_vgg16_model_trained_2/'+model_savepath)   


# In[13]:


# --------------------------------------
# Affichage des courbes accuracy et Loss
# --------------------------------------
if (TRAIN):
    plt.figure(1)
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show() 


# In[14]:


# Evaluation du modèle
(eval_loss, eval_accuracy) = net_final.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_TEST)
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# In[15]:


# Affichage des classes du jeu d'entrainement
train_generator.class_indices


# In[16]:


# Génération des prédictions
test_generator.reset()
pred = net_final.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)


# In[17]:


# La colonne 1 représente la probabilité que l'image soit un chien (la colonne 0 que l'image soit un chat)
pred[:,1]


# In[18]:


# Enregistrement fichier soumission
soumission=pd.DataFrame({"id":test_generator.filenames,"label":pred[:,1]})
soumission['id'] = soumission['id'].str[:-4].astype('int')
soumission = soumission.sort_values(by=['id'])
soumission.to_csv('soumission.csv', index=False)


# In[19]:


# ---------------------------------------------------------
# Affichage aléatoires images prédites 0 = chat / 1 = chien
# ---------------------------------------------------------
import random

n = soumission.shape[0]
f = list(np.arange(1,n))

c = 20
r =random.sample(f, c)
nrows = 4
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*5, ncols*5))    
for i in range(c):
    file = str(soumission['id'][r[i]]) + '.jpg'
    path = TESTING_DIR + "/" + file
    img = plt.imread(path)
    plt.subplot(4, 5, i+1)
    plt.imshow(img, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title(str(soumission['id'][r[i]])+"\n"+str(soumission['label'][r[i]]))
plt.show()

