#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np

from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.utils import class_weight as cw
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Flatten, Dense, Dropout

# Import des librairies pour la gestion des Callback
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping


# In[2]:


EPOCHS                  = 100   # Nombre d'epoch
IMGSIZE                 = 96    # Taille des images
BATCH_SIZE              = 32    # Pour le traitement par lot des images (optimisation de la decente de gradient)
STOPPING_PATIENCE       = 10    # Callback pour stopper si le modèle n'apprend plus
VERBOSE                 = 0     # Niveau de verbosité
MODEL_NAME              = 'cnn_80epochs_imgsize160'
OPTIMIZER               = 'adam'
TRAINING_DIR            = '../input/dogs-vs-cats-redux-kernels-edition/train'
TEST_DIR                = '../input/dogs-vs-cats-redux-kernels-edition/test'
TRAIN_MODEL             = True  # Entrainement du modele (True) ou chargement (False)


# In[3]:


train_files = os.listdir(TRAINING_DIR)
train_labels = []

for file in train_files:
    train_labels.append(file.split(".")[0])
    
df_train = pd.DataFrame({"id": train_files, "label": train_labels})

df_train.head()


# In[4]:


# Augmentation d'images à la volée et split train / validation
train_datagen =          ImageDataGenerator(
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.3,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            validation_split=0.10)

# Parcours du jeu d'entrainement (subset = 'training')
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


# In[5]:


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


# In[6]:


test_files = os.listdir(TEST_DIR)
df_test = pd.DataFrame({"id": test_files, 'label': 'nan'})


# In[7]:


# https://medium.com/@vijayabhaskar96/tutorial-on-keras-flow-from-dataframe-1fd4493d237c
# Le ImageDataGenerator fait juste une normalisation des valeurs
test_datagen = ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_dataframe(
    df_test, 
    TEST_DIR, 
    x_col='id',
    y_col=None,       # None car nous ne connaissons pas les labels
    has_ext=True, 
    target_size=(IMGSIZE, IMGSIZE), 
    class_mode=None,  # None pour le jeu de test
    seed=42,
    batch_size=1,     # batch_size = 1 sur le jeu de test
    shuffle=False     # Pas de mélange sur le jeu de test
)


# In[8]:


# Cette fonction permet de retourner le ratio entre chat vs chien (utile dans le cas ou une classe et proéminente sur les autres)
def get_weight(y):
    class_weight_current =  cw.compute_class_weight('balanced', np.unique(y), y)
    return class_weight_current
class_weights = get_weight(train_generator.classes)


# In[9]:


# Génération des STEPS_SIZE (comme nous utilisons des générateurs infinis)
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
STEP_SIZE_TEST  = test_generator.n  // test_generator.batch_size


# In[10]:


# Permet de stopper l'apprentissage si il stagne
EARLY_STOPPING =         EarlyStopping(
            monitor='val_loss',
            patience=STOPPING_PATIENCE,
            verbose=VERBOSE,
            mode='auto')


# Reduit le LearningRate si stagnation
LR_REDUCTION =         ReduceLROnPlateau(
            monitor='val_acc',
            patience=5,
            verbose=VERBOSE,
            factor=0.5,
            min_lr=0.00001)

CALLBACKS = [EARLY_STOPPING, LR_REDUCTION]


# In[11]:


# Initialisation du modèle
classifier = Sequential()

# Réalisation des couches de Convolution  / Pooling

# ---- Conv / Pool N°1
classifier.add(Conv2D(filters=16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      input_shape=(IMGSIZE, IMGSIZE, 3),
                      activation='relu'))

classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# ---- Conv / Pool N°2
classifier.add(Conv2D(filters=16,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu'))

classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# ---- Conv / Pool N°3
classifier.add(Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu'))

classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# ---- Conv / Pool N°4
classifier.add(Conv2D(filters=32,
                      kernel_size=3,
                      strides=1,
                      padding='same',
                      activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size=(2, 2), strides=2))


# Fully Connected
# Flattening : passage de matrices 3D vers un vecteur
classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.1))


# Couche de sortie : classification => softmax sur le nombre de classe
classifier.add(
    Dense(
        units=2,
        activation='softmax',
        name='softmax'))

# compilation du  model de classification
classifier.compile(
    optimizer=OPTIMIZER,
    loss='categorical_crossentropy',
    metrics=['accuracy'])


print("Input Shape :{}".format(classifier.get_input_shape_at(0)))
classifier.summary()


# In[12]:


def train_model():
    # https://keras.io/models/sequential/#fit_generator
    # Pour visualisation avec Tensorboard (console anaconda): 
    # tensorboard --logdir=/full_path_to_your_logs
    history = classifier.fit_generator(
        generator=train_generator,           # le générateur pour les données d'entrainement
        steps_per_epoch=STEP_SIZE_TRAIN,     # le Step_size pour les données d'entrainement
        validation_data=valid_generator,     # le générateur pour les données de validation
        validation_steps=STEP_SIZE_VALID,    # le Step_size pour les données de validation
        epochs=EPOCHS,                       # le nombre d'epoch sur l'ensemble du jeu de données
        verbose=VERBOSE,                     # la verbosité
        class_weight=class_weights,          # le ratio de répartition des classes chien/chat
        callbacks=CALLBACKS)                 # la liste des fonctions de callback à appeler après chaque epoch
    return history    


# In[13]:


def plot_history(history):
    # --------------------------------------
    # Affichage des courbes accuracy et Loss
    # --------------------------------------
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


if (TRAIN_MODEL):
    print("Entrainement du modèle CNN")
    hist = train_model()     # Entrainement du modèle
    plot_history(hist)       # Affichage de la courbe d'apprentissage
    classifier.save(MODEL_NAME + '.h5')
else:
    print("Chargement du modèle...")
    classifier = load_model('../input/weight/cnn/cnn.h5')


# In[15]:


classifier.evaluate_generator(generator=valid_generator, steps=STEP_SIZE_TEST)


# In[16]:


# Le générateur doit être reseter avant utilisation pour les prédictions
test_generator.reset()
pred=classifier.predict_generator(test_generator, steps=STEP_SIZE_TEST, verbose=1)


# In[17]:


# Visualisation du vecteur de probabilité des 5 premières lignes des prédictions
pred[0:5,:]


# In[18]:


predicted_class_indices=np.argmax(pred,axis=1)
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[19]:


# Création d'un dataframe contenant les images et classes prédites
filenames=test_generator.filenames
results=pd.DataFrame({"id":filenames,"label":predictions})
results.head()


# In[20]:


# copy du dataframe de resultat
soumission = results.copy()

# suppression de l'extension du fichier et conversion de la colonne en int avec la méthode vectorielle str
soumission['id'] = soumission['id'].str[:-4].astype('int')
soumission.head()


# In[21]:


# Tri sur la colonne des id avec la methode sort_values du dataframe
soumission = soumission.sort_values(by=['id'])
soumission.head()


# In[22]:


# Remplacement du label 'cat' ou 'dog' par une valeur numérique : utilisation de la fonction replace
# Rappel sur les classes : {0: "Cat", 1: "Dog"} 
soumission.replace({'dog': 1, 'cat': 0}, inplace=True)
soumission.head()


# In[23]:


# conversion du Dataframe vers un fichier de sortie
# This is saved in the same directory as your notebook
filename = 'results.csv'
soumission.to_csv(filename,index=False)
print('Fichier enregistré: ' + filename)


# In[24]:


import random

n = results.shape[0]
f = list(np.arange(1,n))

c = 20
r =random.sample(f, c)
nrows = 4
ncols = 5
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(nrows*5, ncols*5))    
for i in range(c):
    file = str(results['id'][r[i]])
    path = TEST_DIR+"/"+file
    img = plt.imread(path)
    plt.subplot(4, 5, i+1)
    plt.imshow(img, aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.title(str(results['id'][r[i]])+"\n"+str(results['label'][r[i]]))
plt.show()

