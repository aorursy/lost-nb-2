#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("../input/nnfl-cnn-lab2/upload"))


# In[ ]:


FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[ ]:


import io
df = pd.read_csv('../input/nnfl-cnn-lab2/upload/train_set.csv')


# In[ ]:


df = df.replace(0,'buildings').replace(1,'forest').replace(2,'glacier').replace(3,'mountain').replace(4,'sea').replace(5,'street')
df.head()


# In[ ]:


df.tail()


# In[ ]:


df['label'].value_counts().plot.bar()


# In[ ]:


filenames = os.listdir("../input/nnfl-cnn-lab2/upload/train_images/train_images")
sample = random.choice(filenames)
image = load_img("../input/nnfl-cnn-lab2/upload/train_images/train_images/" + sample)
plt.imshow(image)


# In[ ]:


import tensorflow as tf
tf.keras.backend.clear_session()


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(32, activation='relu'))


# In[ ]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.summary()


# In[ ]:


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# In[ ]:


earlystop = EarlyStopping(patience=10)


# In[ ]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[ ]:


callbacks = [earlystop, learning_rate_reduction]


# In[ ]:


df["label"] = df["label"].replace({0: 'type0', 1: 'type1', 2: 'type2', 3: 'type3', 4: 'type4', 5: 'type5'})
df['label'].head()


# In[ ]:


train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[ ]:


train_df['label'].value_counts().plot.bar()


# In[ ]:


validate_df['label'].value_counts().plot.bar()


# In[ ]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# In[ ]:


print(total_train)
print(total_validate)


# In[ ]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "../input/nnfl-cnn-lab2/upload/train_images/train_images", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[ ]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "../input/nnfl-cnn-lab2/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[ ]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "../input/nnfl-cnn-lab2/upload/train_images/train_images/", 
    x_col='image_name',
    y_col='label',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[ ]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[ ]:


epochs=6 if FAST_RUN else 6
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)f


# In[ ]:


model.save_weights("model.h5")


# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[ ]:


test_filenames = os.listdir("../input/nnfl-cnn-lab2/upload/test_images/test_images/")
test_df = pd.DataFrame({
    'image_name': test_filenames
})
nb_samples = test_df.shape[0]
test_df.head()


# In[ ]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "../input/nnfl-cnn-lab2/upload/test_images/test_images/", 
    x_col='image_name',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[ ]:


predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))


# In[ ]:


test_df['label'] = np.argmax(predict, axis=-1)
test_df.head()


# In[ ]:


label_map = dict((v,k) for k,v in train_generator.class_indices.items())
test_df['label'] = test_df['label'].replace(label_map)
test_df.head()


# In[ ]:


test_df["label"] = test_df["label"].replace({'type0': 0, 'type1': 1, 'type2': 2, 'type3': 3, 'type4': 4, 'type5': 5})


# In[ ]:


test_df['label'].value_counts().plot.bar()


# In[ ]:


sample_test = test_df.head(18)
sample_test.head()
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    image_name = row['image_name']
    label = row['label']
    img = load_img("../input/nnfl-cnn-lab2/upload/test_images/test_images/"+image_name, target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(image_name + '(' + "{}".format(label) + ')' )
plt.tight_layout()
plt.show()


# In[ ]:


test_df = test_df.replace('buildings',0).replace('forest',1).replace('glacier',2).replace('mountain',3).replace('sea',4).replace('street',5)
submission_df = test_df.copy()


# In[ ]:


image = load_img("../input/nnfl-cnn-lab2/upload/test_images/test_images/" + "13407.jpg")
plt.imshow(image)


# In[ ]:


submission_df.to_csv('submission.csv', index=False)
submission_df.head()


# In[ ]:


df=pd.read_csv('/kaggle/working/submission.csv')
from IPython.display import HTML
import pandas as pd
import numpy as np
import base64
def create_download_link(df, title = "Download CSV file", filename = "data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}"target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)
create_download_link(df)


# In[ ]:




