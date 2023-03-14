%tensorflow_version 1.x
import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from keras.models import load_model, Model
import matplotlib.pyplot as plt
import random
import os
from datetime import datetime
import re

!pip install -U -q kaggle --force

from google.colab import files
f = files.upload()

!mkdir -p ~/.kaggle

!cp kaggle.json ~/.kaggle/
!chmod 600 /root/.kaggle/kaggle.json
!kaggle competitions download -c nnfl-cnn-lab2


%%bash
cd /content
unzip nnfl-cnn-lab2.zip

print(os.listdir("../content"))

train_filenames = os.listdir("../content/upload/train_images/train_images")
train_folder = "../content/upload/train_images/train_images/"
print(train_filenames)

finalPre_filenames = os.listdir("../content/upload/test_images/test_images")
finalPre_folder = "../content/upload/test_images/test_images/"
print(finalPre_filenames)

df = pd.read_csv('../content/upload/train_set.csv')

df['label'].value_counts()

sample = random.choice(train_filenames)
image = load_img(train_folder+sample)
plt.imshow(image)

print(image.mode)
print(image.size)

df['label'] = df['label'].astype(str)


IMAGE_WIDTH=150
IMAGE_HEIGHT=150
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

train_df, test_df = train_test_split(df, test_size=0.1)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_df, validate_df = train_test_split(train_df, test_size=0.1))
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

print(train_df['label'].value_counts())
print(test_df['label'].value_counts())
print(validate_df['label'].value_counts())

batch_size = 10



train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range = 0.2,
    height_shift_range = 0.2
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    train_folder, 
    x_col='image_name',
    y_col="label",
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)



validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    train_folder,
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)



test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_dataframe(
    test_df, 
    train_folder,
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, PReLU, LeakyReLU
from keras.regularizers import l2
from keras.layers import  concatenate
from keras.layers import Input, ZeroPadding2D


in1 = Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))

conv1a = Conv2D(16, (1, 1), padding='same', activation='relu')(in1)
conv1b = Conv2D(16, (3, 3), padding='same', activation='relu')(in1)
conv1c = Conv2D(16, (5, 5), padding='same', activation='relu')(in1)
conv1po1 = Conv2D(16, (1, 1), activation='relu')(in1)
conv1pool = MaxPooling2D(3, strides=1, padding='same')(conv1po1)
conv1 = concatenate([conv1a, conv1b, conv1c,conv1pool ], axis=-1)
conv1 = BatchNormalization()(conv1)
conv1 = Dropout(0.3)(conv1)

conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = MaxPooling2D(pool_size=(3, 3))(conv2)
conv2 = Dropout(0.3)(conv2)

conv3 = Conv2D(128, (3, 3), activation='relu')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = AveragePooling2D(pool_size=(2, 2))(conv3)
conv3 = Dropout(0.25)(conv3)

flattened = Flatten()(conv3)

dense1 = Dense(128, activation='relu')(flattened)
dense1 = BatchNormalization()(dense1)
dense1 = Dropout(0.3)(dense1)

dense2 = Dense(128)(dense1)
dense2 = PReLU()(dense2)
dense2 = BatchNormalization()(dense2)
dense2 = Dropout(0.4)(dense2)

dense3 = Dense(128)(dense2)
dense3 = PReLU()(dense3)
dense3 = BatchNormalization()(dense3)
dense3 = Dropout(0.5)(dense3)

output_layer1 = Dense(6, activation='softmax')(dense3)

model = Model(inputs=in1, outputs=output_layer1)


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.3, 
                                            min_lr=0.00001)

checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_acc', verbose=1,
    save_best_only=True, mode='auto', period=1)

callbacks = [earlystop,
            #  learning_rate_reduction, 
             checkpoint]

total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
total_test = test_df.shape[0]
print(total_train)
print(total_validate)
print(total_test)

epochs=30
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


model.save_weights("model.h5")

best_model = load_model('best_model.hdf5')

prediction_mod = model.predict_generator(test_generator, steps=np.ceil(total_test/batch_size))
prediction_best_mod = best_model.predict_generator(test_generator, steps=np.ceil(total_test/batch_size))


test_df['labelp_m'] = np.argmax(prediction_mod, axis=-1)
test_df['labelp_bm'] = np.argmax(prediction_best_mod, axis=-1)

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['labelp_m'] = test_df['labelp_m'].replace(label_map)
test_df['labelp_bm'] = test_df['labelp_bm'].replace(label_map)

test_df

from sklearn.metrics import accuracy_score
print(accuracy_score(test_df['label'].astype(int), test_df['labelp_m'].astype(int)))
print(accuracy_score(test_df['label'].astype(int), test_df['labelp_bm'].astype(int)))

#To sort the filenames, in directory order
import re
finalPre_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

finalPre_df = pd.DataFrame({
    'image_name': finalPre_filenames
})
total_finalPre = finalPre_df.shape[0]

finalPre_datagen = ImageDataGenerator(rescale=1./255)
finalPre_generator = finalPre_datagen.flow_from_dataframe(
    finalPre_df, 
    finalPre_folder,
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)

finalPrediction = model.predict_generator(finalPre_generator, steps=np.ceil(total_finalPre/batch_size))

finalPre_df['label'] = np.argmax(finalPrediction, axis=-1)

finalPre_df

label_map = dict((v,k) for k,v in train_generator.class_indices.items())
finalPre_df['label'] = finalPre_df['label'].replace(label_map)

finalPre_df['label'].value_counts()

finalPre_df.to_csv('submission.csv', index=False)

!kaggle competitions submit -c 'nnfl-cnn-lab2' -f submission.csv -m "fail7"

!kaggle competitions submissions -c nnfl-cnn-lab2

!kaggle competitions leaderboard -c nnfl-cnn-lab2 --show
