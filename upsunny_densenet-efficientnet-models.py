#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install -q efficientnet')


# In[2]:


#P7.17 导入库
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.applications import DenseNet121
import efficientnet.tfkeras as efn


# In[3]:


#P7.18 读取数据集，划分训练集和验证集
EPOCHS = 20
SAMPLE_LEN = 100
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)


# In[4]:


train_data.head()


# In[5]:


test_data.head()


# In[6]:


AUTO = tf.data.experimental.AUTOTUNE
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.experimental.TPUStrategy(tpu)

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
GCS_DS_PATH = KaggleDatasets().get_gcs_path()


# In[7]:


def format_path(image_id):
    return  GCS_DS_PATH + '/images/' + image_id + '.jpg'
train_paths = train_data.image_id.apply(format_path).values
train_labels = np.float32(train_data.loc[:, 'healthy':'scab'].values)
test_paths = test_data.image_id.apply(format_path).values
train_paths, valid_paths, train_labels, valid_labels = train_test_split(                train_paths, train_labels, test_size=0.3, random_state=2020)


# In[8]:


#P7.19 图像数据加载与解码函数
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

#P7.20 数据增强函数
def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label


# In[9]:


#P7.21 构建训练集、验证集和测试集
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# In[10]:


#P7.22 定义学习率动态调度函数
def build_lrfn(lr_start=0.00001, lr_max=0.00005, 
               lr_min=0.00001, lr_rampup_epochs=5, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) *                 lr_exp_decay**(epoch - lr_rampup_epochs                                - lr_sustain_epochs) + lr_min
        return lr
    return lrfn
lrfn = build_lrfn()
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)


# In[11]:


#P7.23下载并重定义DenseNet121迁移模型模型
with strategy.scope():
    model = tf.keras.Sequential([DenseNet121(input_shape=(512, 512, 3),
                                             weights='imagenet',
                                             include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
        
    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()


# In[12]:


get_ipython().run_cell_magic('time', '', '#P7.24 DenseNet121模型训练\nEPOCHS=20\nhistory = model.fit(train_dataset,\n                    epochs=EPOCHS,\n                    callbacks=[lr_schedule],\n                    steps_per_epoch=STEPS_PER_EPOCH,\n                    validation_data=valid_dataset)')


# In[13]:


#P7.26 绘制准确率曲线
display_training_curves(
    history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'], 
    'accuracy')


# In[14]:


get_ipython().run_cell_magic('time', '', '#P7.25 绘制模型的学习曲线\ndef display_training_curves(training, validation, yaxis):\n    if yaxis == "loss":\n        ylabel = "Loss"\n        title = "Loss vs. Epochs"\n    else:\n        ylabel = "Accuracy"\n        title = "Accuracy vs. Epochs"        \n    fig = go.Figure()        \n    fig.add_trace(\n        go.Scatter(x=np.arange(1, EPOCHS+1), mode=\'lines+markers\', y=training, marker=dict(color="dodgerblue"),\n               name="Train"))    \n    fig.add_trace(\n        go.Scatter(x=np.arange(1, EPOCHS+1), mode=\'lines+markers\', y=validation, marker=dict(color="darkorange"),\n               name="Val"))    \n    fig.update_layout(title_text=title, yaxis_title=ylabel, xaxis_title="Epochs")\n    fig.show()\n    ')


# In[15]:


#P7.27 用DenseNet模型对四种标签做抽样预测并分析结果

def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(IMAGE_PATH + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

train_images = train_data["image_id"][:4].progress_apply(load_image)  # 前四幅图像

def process(img):
    return cv2.resize(img/255.0, (512, 512)).reshape(-1, 512, 512, 3)
def predict(img):
    return model.layers[2](model.layers[1](model.layers[0](process(img)))).numpy()[0]

def displayResult(img, preds, title):
    fig = make_subplots(rows=1, cols=2)
    colors = {"Healthy":px.colors.qualitative.Plotly[0], "Scab":px.colors.qualitative.Plotly[0],
              "Rust":px.colors.qualitative.Plotly[0], "Multiple diseases":px.colors.qualitative.Plotly[0]}
    if list.index(preds.tolist(), max(preds)) == 0:
        pred = "Healthy"
    if list.index(preds.tolist(), max(preds)) == 1:
        pred = "Scab"
    if list.index(preds.tolist(), max(preds)) == 2:
        pred = "Rust"
    if list.index(preds.tolist(), max(preds)) == 3:
        pred = "Multiple diseases"

    colors[pred] = px.colors.qualitative.Plotly[1]
    colors["Healthy"] = "seagreen"
    colors = [colors[val] for val in colors.keys()]
    fig.add_trace(go.Image(z=cv2.resize(img, (205, 136))), row=1, col=1)
    fig.add_trace(go.Bar(x=["Healthy", "Multiple diseases", "Rust", "Scab"], 
                         y=preds, marker=dict(color=colors)), row=1, col=2)
    fig.update_layout(height=400, width=800, title_text=title, showlegend=False)
    fig.show()
    
preds = predict(train_images[2])
displayResult(train_images[2], preds, "DenseNet Predictions")

preds = predict(train_images[0])
displayResult(train_images[0], preds, "DenseNet Predictions")

preds = predict(train_images[3])
displayResult(train_images[3], preds, "DenseNet Predictions")

preds = predict(train_images[1])
displayResult(train_images[1], preds, "DenseNet Predictions")


# In[16]:


get_ipython().run_cell_magic('time', '', "#P7.28 对测试集做预测，保存预测结果\nprobs_densenet = model.predict(test_dataset, verbose=1)\nsub.loc[:, 'healthy':] = probs_densenet\nsub.to_csv('submission_densenet.csv', index=False)\nsub.head()")


# In[17]:


#P7.29 下载并定义EfficientNetB7模型
with strategy.scope():
    model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                    weights='imagenet',
                                                    include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
    
    
        
    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()


# In[18]:


#P7.30 EfficientNetB7模型训练
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


# In[19]:


#P7.31 绘制EfficientNetB7模型准确率曲线
display_training_curves(
    history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'], 
    'accuracy')


# In[20]:


#P7.32 EfficientNetB7模型抽样检测
preds = predict(train_images[2])
displayResult(train_images[2], preds, "EfficientNetB7 Predictions")

preds = predict(train_images[0])
displayResult(train_images[0], preds, "EfficientNetB7 Predictions")

preds = predict(train_images[3])
displayResult(train_images[3], preds, "EfficientNetB7 Predictions")

preds = predict(train_images[1])
displayResult(train_images[1], preds, "EfficientNetB7 Predictions")


# In[21]:


probs_efnB7 = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_efnB7
sub.to_csv('submission_efnB7.csv', index=False)
sub.head()


# In[22]:


#P7.34 定义EfficientNet NoisyStudent模型
with strategy.scope():
    model = tf.keras.Sequential([efn.EfficientNetB7(input_shape=(512, 512, 3),
                                                    weights='noisy-student',
                                                    include_top=False),
                                 L.GlobalAveragePooling2D(),
                                 L.Dense(train_labels.shape[1],
                                         activation='softmax')])
    
    
        
    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    model.summary()


# In[23]:


#P7.35 模型训练
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    callbacks=[lr_schedule],
                    steps_per_epoch=STEPS_PER_EPOCH,
                    validation_data=valid_dataset)


# In[24]:


#P7.36 绘制准确率曲线
display_training_curves(
    history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'], 
    'accuracy')


# In[25]:


# 抽样检测
preds = predict(train_images[2])
displayResult(train_images[2], preds, " Noisy Student Predictions")

preds = predict(train_images[0])
displayResult(train_images[0], preds, " Noisy Student Predictions")

preds = predict(train_images[3])
displayResult(train_images[3], preds, " Noisy Student Predictions")

preds = predict(train_images[1])
displayResult(train_images[1], preds, " Noisy Student Predictions")


# In[26]:


# P7.37 保存测试集预测结果
probs_efnns = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs_efnns
sub.to_csv('submission_efnns.csv', index=False)
sub.head()


# In[27]:


#P7.38 模型集成
ensemble_1, ensemble_2, ensemble_3 =[sub]*3
#集成模型1
ensemble_1.loc[:, 'healthy':] = 0.50*probs_efnB7 + 0.50*probs_densenet 
ensemble_1.to_csv('submission_ensemble_1.csv', index=False)
#集成模型2
ensemble_2.loc[:, 'healthy':] = 0.25*probs_efnB7 + 0.75*probs_densenet
ensemble_2.to_csv('submission_ensemble_2.csv', index=False)
#集成模型3
ensemble_3.loc[:, 'healthy':] = 0.75*probs_efnB7 + 0.25*probs_densenet
ensemble_3.to_csv('submission_ensemble_3.csv', index=False)
#显示集成模型1
ensemble1 = pd.read_csv('submission_ensemble_1.csv')
ensemble1.head()


# In[28]:


#P7.39 集成模型1预测的最大概率值分布
import matplotlib.pyplot as plt
model1 = ensemble1.drop('image_id',axis = 1)
label_values1 = [np.max(model1.loc[i]) for i in range(1821)]
x=range(1821)
plt.figure(figsize=(5,5))
plt.scatter(x,label_values1)
plt.xlabel('test_id',size=16)
plt.ylabel('Maximum probability',size=16)
plt.show()
print('概率值低于0.5的样本数量为：{0}'.format(np.sum(np.array(label_values1)<0.5)))


# In[29]:


print('概率值大于0.9的样本数量为：{0}'.format(np.sum(np.array(label_values1)>=0.9)))


# In[30]:


1428/1821


# In[31]:


ensemble1 = pd.read_csv('submission_ensemble_1.csv')
ensemble1.head()


# In[ ]:




