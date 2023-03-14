#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import csv
import gc
import cv2
from tqdm import tqdm_notebook

train_csv = "../input/aptos2019-blindness-detection/train.csv"
test_csv = "../input/aptos2019-blindness-detection/test.csv"
train_dir = "../input/aptos2019-blindness-detection/train_images/"
test_dir = "../input/aptos2019-blindness-detection/test_images/"


# In[2]:


df_train = pd.read_csv(train_csv) 
size = 256,256 # input image size
df_test = pd.read_csv(test_csv)
NUM_CLASSES = df_train['diagnosis'].nunique()
print(NUM_CLASSES)


# In[3]:


# cropping function (uses edge detection to crop images)
def get_cropped_image(image):
    img = cv2.blur(image,(2,2))
    #gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    slice1Copy = np.uint8(img)
    canny = cv2.Canny(slice1Copy, 0, 50)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    cropped_img = img[y1:y2, x1:x2]
    cropped_img = cv2.resize(cropped_img, size)
    return cropped_img


# In[4]:


'''sample_to_show = ['07419eddd6be.png','0124dffecf29.png']

def get_cropped_image_demo(image):
    img = cv2.blur(image,(2,2))
    slice1Copy = np.uint8(img)
    canny = cv2.Canny(slice1Copy, 0, 50)
    pts = np.argwhere(canny>0)
    y1,x1 = pts.min(axis=0)
    y2,x2 = pts.max(axis=0)
    cropped_img = img[y1:y2, x1:x2]
    return np.array(cropped_img)

names = []
samples = []
cropped_images = []
for i in sample_to_show:
    path = train_dir + str(i)
    img_ = cv2.imread(path)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    samples.append(img_)
    cropped_ = get_cropped_image_demo(img_)
    cropped_images.append(cropped_)
    
fig = plt.figure(figsize = (5,5))
ax1 = fig.add_subplot(2,2,1)
ax1.title.set_text('original image'), ax1.axis("off"), plt.imshow(samples[0])
ax2 = fig.add_subplot(2,2,2)
ax2.title.set_text('cropped image'), ax2.axis("off"), plt.imshow(cropped_images[0])
ax3 = fig.add_subplot(2,2,3)
ax3.title.set_text('original image'), ax3.axis("off"), plt.imshow(samples[1])
ax4 = fig.add_subplot(2,2,4)
ax4.title.set_text('cropped image'), ax4.axis("off"), plt.imshow(cropped_images[1]);'''


# In[5]:


def load_image(path):
    img=cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(cv2.imread(path), size)
    img = get_cropped_image(img)
    return img


# In[6]:


import matplotlib.pyplot as plt
import PIL     #df.at[2, 'StartDate']

img1 = load_image(train_dir+str(df_train.at[0,"id_code"])+str(".png"))
img2 = load_image(train_dir+str(df_train.at[1,"id_code"])+str(".png"))

#fig, ax = plt.subplots(nrows=2, ncols=2)

plt.figure(1)
plt.subplot(211)
plt.imshow(img1)

plt.subplot(212)
plt.imshow(img2)
plt.show()
 


# In[7]:


training_paths = [train_dir + str(x) + str(".png") for x in df_train["id_code"]]
images = np.empty((len(df_train), 256,256,3), dtype = np.uint8)
for i, path in tqdm_notebook(enumerate(training_paths)):
    images[i,:,:,:] = load_image(path)


# In[8]:


print(len(images))
#plt.show()


# In[9]:


labels = df_train["diagnosis"].values.tolist()
labels = keras.utils.to_categorical(labels)


# In[10]:


images, x_val, labels, y_val = train_test_split(images, labels, test_size = 0.15)


# In[11]:


images, x_test, labels, y_test = train_test_split(images, labels, test_size = 0.08,shuffle=False)


# In[12]:


train_aug = ImageDataGenerator(horizontal_flip = True,
                               zoom_range = 0.25,
                               rotation_range = 60,
                               vertical_flip = True,
                              shear_range=0.1)

train_generator = train_aug.flow(images, labels, batch_size = 8)


# In[13]:


val_aug=train_aug.flow(x_val,y_val,batch_size=8)


# In[14]:


test_aug=train_aug.flow(x_test,y_test,batch_size=8)


# In[15]:


'''input_layer = Input(shape = (256,256,3))
base_model = DenseNet121(include_top = False, input_tensor = input_layer, weights = "../input/densenet-keras/DenseNet-BC-121-32-no-top.h5")
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
out = Dense(5, activation = 'softmax')(x)

model = Model(inputs = input_layer, outputs = out)


# In[16]:


BATCH_SIZE = 32
EPOCHS = 20
WARMUP_EPOCHS = 2
LEARNING_RATE = 1e-4
WARMUP_LEARNING_RATE = 1e-3
HEIGHT = 256
WIDTH = 256
CANAL = 3
N_CLASSES = 5
ES_PATIENCE = 5
RLROP_PATIENCE = 3
DECAY_DROP = 0.5


# In[17]:


from tensorflow.keras.applications import ResNet50 
def build_model(input_shape, n_out):
    input_tensor = Input(shape=input_shape)
    base_model = applications.ResNet50(weights='imagenet', 
                                       include_top=False,
                                       input_tensor=input_tensor)
    #base_model = ResNet50(include_top = False,weights = 'imagenet',input_tensor = input_tensor)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    final_output = Dense(n_out, activation='softmax', name='final_output')(x)
    model = Model(input_tensor, final_output)
    
    return model


# In[18]:


from keras import optimizers, applications
from keras.models import Model
model = build_model(input_shape=(HEIGHT, WIDTH, CANAL), n_out=N_CLASSES)
model.summary()


# In[19]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
for layer in model.layers:
    layer.trainable = True

es = EarlyStopping(monitor='val_loss', mode='min', patience=ES_PATIENCE, restore_best_weights=True, verbose=1)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience=RLROP_PATIENCE, factor=DECAY_DROP, min_lr=1e-6, verbose=1)

callback_list = [es, rlrop]
optimizer = optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=['accuracy'])


# In[20]:


#kappa_metrics = Metrics()
history = model.fit_generator(
    train_generator,
    steps_per_epoch=images.shape[0] / 8,
    epochs=20,
    validation_data=(x_val, y_val),
    callbacks=callback_list
)


# In[21]:


#model.save('densenet.h5')


# In[22]:


accu = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(accu, label="Accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Acc', 'val_acc'])
plt.plot(np.argmax(history.history["val_acc"]), np.max(history.history["val_acc"]), marker="x", color="r",
         label="best model")
plt.show()


# In[23]:


from tqdm import tqdm
(eval_loss, eval_accuracy) = tqdm(
    model.evaluate_generator(generator=val_aug, steps=201, pickle_safe=False))
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# In[24]:


from tqdm import tqdm
(eval_loss, eval_accuracy) = tqdm(model.evaluate_generator(generator=test_aug, steps=201, pickle_safe=False))
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# In[25]:


from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input
from keras.activations import softmax
from keras.activations import elu
from keras.activations import relu
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from tqdm import tqdm
from keras.layers import LeakyReLU


def create_resnet(img_dim, CHANNEL, n_class):
    input_tensor = Input(shape=(256, 256, 3))
    base_model = ResNet50(include_top=False, input_tensor=input_tensor,weights='../input/resnet50weightsfile/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    #base_model.load_weights('../input/ResNet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    x = GlobalAveragePooling2D()(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(alpha=0.3)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.3)(x)
    #x.add(LeakyReLU(alpha=0.1))
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.3)(x)
    #x.add(LeakyReLU(alpha=0.1))
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    output_layer = Dense(n_class, activation='softmax', name="Output_Layer")(x)
    model_resnet = Model(input_tensor, output_layer)

    return model_resnet


model_resnet = create_resnet(256, 3, NUM_CLASSES)


# In[26]:


for layers in model_resnet.layers:
    layers.trainable = True


# In[27]:


lr = 1e-3
optimizer = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True) 

es = EarlyStopping(monitor='val_loss', mode='min', patience = 5, restore_best_weights = True)
rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', patience = 2, factor = 0.5, min_lr=1e-6)
    
callback_list = [es, rlrop]

model_resnet.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["accuracy"]) 


# In[28]:


h=model_resnet.fit_generator(generator = train_generator, steps_per_epoch = len(train_generator), epochs = 20, validation_data = (x_val, y_val), callbacks = callback_list)


# In[29]:


del train_generator, images
gc.collect()


# In[30]:


model_json = model_resnet.to_json()
with open("model2.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model_resnet.save_weights("model2.h5")
print("Saved model to disk")


# In[31]:


h.history.keys()


# In[32]:


accu = h.history['acc']
val_acc = h.history['val_acc']

plt.plot(accu, label="Accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Acc', 'val_acc'])
plt.plot(np.argmax(h.history["val_acc"]), np.max(h.history["val_acc"]), marker="x", color="r",
         label="best model")
plt.show()


# In[33]:


from tqdm import tqdm
(eval_loss, eval_accuracy) = tqdm(
    model.evaluate_generator(generator=val_aug, steps=201, pickle_safe=False))
print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
print("[INFO] Loss: {}".format(eval_loss))


# In[34]:


test_df = pd.read_csv(test_csv)
test_paths = [test_dir + str(x) + str(".png") for x in test_df["id_code"]]
test_images = np.empty((len(test_df), 256,256,3), dtype = np.uint8)
for i, path in tqdm_notebook(enumerate(test_paths)):
    test_images[i,:,:,:] = cv2.resize(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), size)


# In[35]:


predprobs = model_resnet.predict(test_images)


# In[36]:


accu = h.history['acc']
val_acc = h.history['val_acc']

plt.plot(accu, label="Accuracy")
plt.plot(val_acc)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(['Acc', 'val_acc'])
plt.plot(np.argmax(h.history["val_acc"]), np.max(h.history["val_acc"]), marker="x", color="r",
         label="best model")
plt.show()


# In[37]:


predictions = []
for i in predprobs:
    predictions.append(np.argmax(i)) 


# In[38]:


id_code = test_df["id_code"].values.tolist()
subfile = pd.DataFrame({"id_code":id_code, "diagnosis":predictions})
subfile.to_csv('submission.csv',index=False)


# In[39]:


#loaded_model.load_weights("..input/MY model/model2.h5")
#print("Loaded model from disk")


# In[ ]:




