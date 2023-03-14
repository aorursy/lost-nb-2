#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('html', '', "<style>\n@import url('https://fonts.googleapis.com/css?family=Ewert|Roboto&effect=3d');\nspan {font-family:'Roboto'; color:black; text-shadow:4px 4px 4px slategray;}  \ndiv.output_area pre{font-family:'Roboto'; font-size:110%; color:#ff355e;}      \n</style>")


# In[2]:


import numpy as np,pandas as pd
import os,h5py,warnings,urllib
import tensorflow as tf,pylab as pl
from tensorflow import image as timg
import tensorflow.keras.layers as tfkl
import tensorflow.keras.applications as tfka
import tensorflow_hub as th
from sklearn.metrics import confusion_matrix,classification_report
from IPython.core.magic import register_line_magic
warnings.filterwarnings('ignore')
pl.style.use('seaborn-whitegrid')
style_dict={'background-color':'silver','color':'#ff355e', 
            'border-color':'white','font-family':'Roboto'}
fpath='../input/quick-draw-images-from-key-points/'
fpath2='../input/quick-draw-images-from-key-points-2/'
fpath3='../input/quick-draw-images-from-key-points-3/'
fpath4='../input/quick-draw-images-from-key-points-4/'
fpath5='../input/quick-draw-images-from-key-points-5/'
fpath6='../input/quick-draw-images-from-key-points-6/'
fpath7='../input/quick-draw-images-from-key-points-7/'
files=sorted(os.listdir(fpath))
files2=sorted(os.listdir(fpath2))
files3=sorted(os.listdir(fpath3))
files4=sorted(os.listdir(fpath4))
files5=sorted(os.listdir(fpath5))
files6=sorted(os.listdir(fpath6))
files7=sorted(os.listdir(fpath7))
files2=files2[1:]+[files2[0]]


# In[3]:


labels=os.listdir('../input/quickdraw-doodle-recognition/'+                  'train_simplified/')
labels=np.array(sorted([l[:-4] for l in labels]))


# In[4]:


D=400; x=[]; y=[]
@register_line_magic
def load_img(n):
    global D,x,y
    if n=='1': fp=fpath; fns=files; m=5
    if n=='2': fp=fpath2; fns=files2; m=5
    if n=='3': fp=fpath3; fns=files3; m=5
    if n=='4': fp=fpath4; fns=files4; m=5
    if n=='5': fp=fpath5; fns=files5; m=5
    if n=='6': fp=fpath6; fns=files6; m=5
    if n=='7': fp=fpath7; fns=files7; m=4
    for i in range(m):
        f=h5py.File(fp+fns[i],'r')
        keys=list(f.keys())
        x+=[f[keys[0]][i*10000:i*10000+D] 
            for i in range(10)]
        y+=[f[keys[1]][i*10000:i*10000+D]
            for i in range(10)]


# In[5]:


get_ipython().run_line_magic('load_img', '1')
get_ipython().run_line_magic('load_img', '2')
get_ipython().run_line_magic('load_img', '3')
get_ipython().run_line_magic('load_img', '4')
get_ipython().run_line_magic('load_img', '5')
get_ipython().run_line_magic('load_img', '6')
get_ipython().run_line_magic('load_img', '7')


# In[6]:


img_size=96
x=np.array(x)
num_classes=x.shape[0]
x=x.reshape(num_classes*D,img_size,img_size,1)
x=tf.convert_to_tensor(x,dtype=tf.uint8)
x=timg.grayscale_to_rgb(x).numpy()
y=np.array(y).reshape(num_classes*D)
N=y.shape[0]; n=int(.1*N)
shuffle_ids=np.arange(N)
np.random.RandomState(12).shuffle(shuffle_ids)
x,y=x[shuffle_ids],y[shuffle_ids]
x.shape,y.shape


# In[7]:


nn=np.random.randint(0,N,7)
ll=[labels[y[nn[i]]] for i in range(7)]
pl.figure(figsize=(10,2))
for i in range(7):
    pl.subplot(1,7,i+1)
    pl.imshow(x[nn[i]])
pl.suptitle('Key Points to Lines: \n%s'%ll);


# In[8]:


x_test,x_valid,x_train=x[:n],x[n:2*n],x[2*n:]
y_test,y_valid,y_train=y[:n],y[n:2*n],y[2*n:]
del x,y,shuffle_ids


# In[9]:


def premodel(pix,den,mh,lbl,activ,loss):
    model=tf.keras.Sequential([
        tf.keras.layers.Input((pix,pix,3),
                              name='input'),
        th.KerasLayer(mh,trainable=True),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(den,activation='relu'),
        tf.keras.layers.Dropout(rate=.5),
        tf.keras.layers.Dense(lbl,activation=activ)])
    model.compile(optimizer='adam',
                  metrics=['accuracy'],loss=loss)
    display(model.summary())
    return model
def cb(fw):
    early_stopping=tf.keras.callbacks    .EarlyStopping(monitor='val_loss',patience=20,verbose=2)
    checkpointer=tf.keras.callbacks    .ModelCheckpoint(filepath=fw,save_best_only=True,verbose=2)
    lr_reduction=tf.keras.callbacks    .ReduceLROnPlateau(monitor='val_loss',verbose=2,
                       patience=5,factor=.8)
    return [checkpointer,early_stopping,lr_reduction]


# In[10]:


fw='weights.best.cv001-%s'%num_classes+'.hdf5'
[handle_base,pixels]=["mobilenet_v2_050_96",img_size]
mhandle="https://tfhub.dev/google/imagenet/{}/feature_vector/4".format(handle_base)


# In[11]:


model=premodel(pixels,2048,mhandle,num_classes,
               'softmax','sparse_categorical_crossentropy')
history=model.fit(x=x_train,y=y_train,batch_size=128,
                  epochs=10,callbacks=cb(fw),
                  validation_data=(x_valid,y_valid))


# In[12]:


model.load_weights(fw)
model.evaluate(x_test,y_test)

