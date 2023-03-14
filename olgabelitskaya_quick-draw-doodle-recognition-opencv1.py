#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display,HTML
def dhtml(str):
    display(HTML("""<style>
    @import 'https://fonts.googleapis.com/css?family=Smokum&effect=3d';      
    </style><h1 class='font-effect-3d' 
    style='font-family:Smokum; color:#aa33ff; font-size:35px;'>
    %s</h1>"""%str))


# In[2]:


dhtml('Code Library, Style, and Links')


# In[3]:


import numpy as np,pandas as pd,keras as ks
import os,ast,cv2,warnings
import pylab as pl
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from keras.models import Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation,Dropout,Dense,Conv2D,MaxPooling2D,GlobalMaxPooling2D
warnings.filterwarnings('ignore')
pl.style.use('seaborn-whitegrid')
style_dict={'background-color':'gainsboro','color':'#aa33ff', 
            'border-color':'white','font-family':'Roboto'}
fpath='../input/quickdraw-doodle-recognition/train_simplified/'
os.listdir("../input")


# In[4]:


dhtml('Data Exploration')


# In[5]:


I=64 # image size in pixels
S=17 # current number of the label set {1,...,17} -> {1-20,..., 321-340}
T=20 # number of labels in one set 
N=24000 # number of images with the same label in the training set
files=sorted(os.listdir(fpath))
labels=[el.replace(" ","_")[:-4] for el in files]
print(labels)


# In[6]:


def display_drawing():
    for k in range(5) :  
        pl.figure(figsize=(10,2))
        pl.suptitle(files[(S-1)*T+k])
        for i in range(5):
            picture=ast.literal_eval(data[labels[(S-1)*T+k]].values[i])
            for x,y in picture:
                pl.subplot(1,5,i+1)
                pl.plot(x,y,'-o',markersize=1,color='slategray')
                pl.xticks([]); pl.yticks([])
            pl.gca().invert_yaxis(); pl.axis('equal');            
def get_image(data,lw=7,time_color=True):
    data=ast.literal_eval(data)
    image=np.zeros((300,300),np.uint8)
    for t,s in enumerate(data):
        for i in range(len(s[0])-1):
            color=255-min(t,10)*15 if time_color else 255
            _=cv2.line(image,(s[0][i]+15,s[1][i]+15),
                       (s[0][i+1]+15,s[1][i+1]+15),color,lw) 
    return cv2.resize(image,(I,I))


# In[7]:


data=pd.DataFrame(index=range(N),
                  columns=labels[(S-1)*T:S*T])
for i in range((S-1)*T,S*T):
    data[labels[i]]=    pd.read_csv(fpath+files[i],
                index_col='key_id').drawing.values[:N]
data.head(3).T.style.set_properties(**style_dict)


# In[8]:


display_drawing()


# In[9]:


images=[]
for label in labels[(S-1)*T:S*T]:
    images.extend([get_image(data[label].iloc[i]) 
                   for i in range(N)])
images=np.array(images,dtype=np.uint8)
targets=np.array([[]+N*[k] for k in range((S-1)*T,S*T)],
                 dtype=np.int32).reshape(N*T)
del data
images.shape,targets.shape


# In[10]:


images=images.reshape(-1,I,I,1)
x_train,x_test,y_train,y_test=train_test_split(images,targets,
                 test_size=.2,random_state=1)
n=int(len(x_test)/2)
x_valid,y_valid=x_test[:n],y_test[:n]
x_test,y_test=x_test[n:],y_test[n:]
del images,targets
[x_train.shape,x_valid.shape,x_test.shape,
 y_train.shape,y_valid.shape,y_test.shape]


# In[11]:


nn=np.random.randint(0,int(.8*T*N),3)
ll=labels[int(y_train[nn[0]])]+   ', '+labels[int(y_train[nn[1]])]+   ', '+labels[int(y_train[nn[2]])]
pl.figure(figsize=(10,2))
pl.subplot(1,3,1); pl.imshow(x_train[nn[0]].reshape(I,I))
pl.subplot(1,3,2); pl.imshow(x_train[nn[1]].reshape(I,I))
pl.subplot(1,3,3); pl.imshow(x_train[nn[2]].reshape(I,I))
pl.suptitle('Key Points to Lines: %s'%ll);


# In[12]:


dhtml('The Model')


# In[13]:


def model():
    model=Sequential()
    model.add(Conv2D(32,(5,5),padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(LeakyReLU(alpha=.02))   
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.2))
    model.add(Conv2D(196,(5,5)))
    model.add(LeakyReLU(alpha=.02))  
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(.2))
    model.add(GlobalMaxPooling2D())   
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=.02))
    model.add(Dropout(.5))   
    model.add(Dense(T))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    return model
model=model()


# In[14]:


print(set(y_train));
print(set(y_train-(S-1)*T))


# In[15]:


fw='weights.best.model.cv321-340.hdf5'
checkpointer=ModelCheckpoint(filepath=fw,verbose=2,
                save_best_only=True)
lr_reduction=ReduceLROnPlateau(monitor='val_loss',
                  patience=5,verbose=2,factor=.75)
history=model.fit(x_train,y_train-(S-1)*T,epochs=100,
                  batch_size=1024,verbose=2,
                  validation_data=(x_valid,y_valid-(S-1)*T),
                  callbacks=[checkpointer,lr_reduction])


# In[16]:


dhtml('Evaluation')


# In[17]:


model.load_weights(fw)
model.evaluate(x_test,y_test-(S-1)*T)


# In[18]:


p_test=model.predict(x_test)
p_test=[np.argmax(x) for x in p_test]
p_test[:10]


# In[19]:


well_predicted=[]
for p in range(len(x_test)):
    if (p_test[p]+(S-1)*T==y_test[p]):
        well_predicted.append(labels[(S-1)*T+p_test[p]])
u=np.unique(well_predicted,return_counts=True)
pd.DataFrame({'labels':u[0],'correct predictions':u[1]}).sort_values('correct predictions',ascending=False).style.set_properties(**style_dict)

