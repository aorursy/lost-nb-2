#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, seaborn as sns, pandas as pd, matplotlib.pyplot as plt, os, cv2, tensorflow as tf, keras, math


# In[2]:


train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')


# In[3]:


size, n_channels = 227, 3
os.chdir('/kaggle/input/plant-pathology-2020-fgvc7/images/')
X = np.zeros((train.shape[0], size, size, n_channels), dtype=np.int16)
for i, image_name in enumerate(train.image_id.values): #this will take sometime to finish
    X[i] = cv2.resize(plt.imread(image_name+'.jpg'), (size, size)).astype(np.int16)


# In[4]:


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 13))
for i, axis in enumerate(ax.flatten()): #these images are from the real training data
    axis.imshow(X[i])
    axis.title.set_text(train.columns[1:][np.argmax(train.iloc[i, 1:].values)])


# In[5]:


images={ 'healthy': [], 'multiple_diseases': [], 'rust': [], 'scab': [] }
       
sub_folders=['Apple___healthy', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___Apple_scab']
for folder in ['color']:#you can add 'segmented' in the list to add more images
    for sub_f in sub_folders:
        os.chdir('/kaggle/input/plantvillage-dataset/'+folder+'/'+sub_f)
        for image_name in os.listdir()[:500]:# remove the slicing [:500] if you want to add all images, here we took 500 images per target
            if 'healthy' in sub_f:
                images['healthy'].append(cv2.resize(plt.imread(image_name), (size, size)).astype(np.int16))
            elif 'rot' in sub_f:
                images['multiple_diseases'].append(cv2.resize(plt.imread(image_name), (size, size)).astype(np.int16))
            elif 'rust' in sub_f:
                images['rust'].append(cv2.resize(plt.imread(image_name), (size, size)).astype(np.int16))
            else:
                images['scab'].append(cv2.resize(plt.imread(image_name), (size, size)).astype(np.int16))


# In[6]:


y = train.iloc[:, 1:].values
target_to_one_hot_vec = {'healthy':np.array([1, 0, 0, 0]), 'multiple_diseases':np.array([0, 1, 0, 0]),
                        'rust':np.array([0, 0, 1, 0]), 'scab':np.array([0, 0, 0, 1])}
total_images=0
for key in ['healthy', 'multiple_diseases', 'rust', 'scab']:
    total_images += len(images[key])
total_images += X.shape[0]
print('Total Number of images is :', total_images)
data_x = np.zeros((total_images, )+X.shape[1:])
data_y = np.zeros((total_images, 4))
data_x[:X.shape[0]] = X.copy()
data_y[:y.shape[0]] = y.copy()
i = X.shape[0]-1
for key in ['healthy', 'multiple_diseases', 'rust', 'scab']:
    for img in images[key]:
        i +=1
        data_x[i] = img
        data_y[i] = target_to_one_hot_vec[key]


# In[7]:


fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 13))
for i, axis in enumerate(ax.flatten()): #these images are from the external dataset
    axis.imshow(images[list(images.keys())[i]][int(np.random.randint(0, 250, 1))])
    axis.title.set_text(list(images.keys())[i])


# In[8]:


reset_selective images #write yes and hit enter when the input box shows up this will remove the variable images from the RAM


# In[9]:


reset_selective X #write yes and hit enter when the input box shows up this will remove the variable X from the RAM


# In[10]:


plt.subplots(figsize=(12, 7))
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Schedule')
lrs= []
for epoch in range(1, 61):
    cos_inner = (math.pi * (epoch % 61)) / (61)
    lrs.append(5e-4/2 * (math.cos(cos_inner) + 1))
sns.lineplot(x=list(range(1, 61)), y=lrs)


# In[11]:


import math
gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=180,
                         zoom_range=.1, width_shift_range=.2, height_shift_range=.2)
mc = keras.callbacks.ModelCheckpoint(filepath='/kaggle/working/model.h5',  verbose=True, save_best_only=True)#For saving the model when the val_loss goes down

def aggressive_lrs(epoch, _):
    cos_inner = (math.pi * (epoch % 61)) / (61)
    return 5e-4/2 * (math.cos(cos_inner) + 1) #initial learning rate is 5e-4
 
lr = keras.callbacks.LearningRateScheduler(aggressive_lrs)


# In[12]:


indices = pd.Series(np.round(np.linspace(0, data_x.shape[0]-1, data_x.shape[0])))
train_indices = indices.sample(3200).values.astype(np.int16)
test_indices = np.array([i for i in indices if i not in train_indices]).astype(np.int16)
#3200 training images, about 400 validation_indices


# In[13]:


os.chdir('/kaggle/working/')
get_ipython().system('git clone https://github.com/qubvel/efficientnet.git')
import efficientnet.efficientnet.keras as efn
md  = efn.EfficientNetB6(weights='imagenet', include_top=False, input_shape=(size, size, 3), pooling='avg')


# In[14]:


model = keras.models.Sequential([md,
                                 keras.layers.Dense(900, activation='relu'),
                                 keras.layers.Dense(800, activation='relu'),
                                 keras.layers.Dense(4, activation='softmax')
                                ])
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


# In[15]:


model.fit_generator(gen.flow(data_x[train_indices], data_y[train_indices], batch_size=8), epochs=60, 
                    steps_per_epoch=111, validation_data=(data_x[test_indices], data_y[test_indices]), 
                    callbacks=[mc, lr])


# In[16]:


reset_selective data_x #write yes and hit enter when the input box shows up this will remove the variable data_x from the RAM


# In[17]:


os.chdir('/kaggle/input/plant-pathology-2020-fgvc7/images/')
X_test = np.zeros((test.shape[0], size, size, n_channels), dtype=np.int16)
for i, image_name in enumerate(test.image_id.values): #this will take sometime to finish
    X_test[i] = cv2.resize(plt.imread(image_name+'.jpg'), (size, size)).astype(np.int16)


# In[18]:


os.chdir('/kaggle/working/')
mdl = keras.models.load_model('model.h5')
y_preds = mdl.predict(X_test)


# In[19]:


os.chdir('/kaggle/input/plant-pathology-2020-fgvc7/')
sb = pd.read_csv('sample_submission.csv')
sb.iloc[:, 1:] = y_preds


# In[20]:


from IPython.display import HTML
import pandas as pd
import numpy as np
import base64

# function that takes in a dataframe and creates a text link to  
# download it (will only work for files < 2MB or so)
def create_download_link(df, title = "Download CSV file", filename = "SubmitMe.csv"):  
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

# create a random sample dataframe
df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))

# create a link to download the dataframe
create_download_link(sb)

# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 


# In[ ]:




