#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd


# In[ ]:


TRAIN_PATH = '../input/dlschool-fashionmnist2/fashion-mnist_train.csv'
TEST_PATH = '../input/dlschool-fashionmnist2/fashion-mnist_test.csv'


# In[ ]:


train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# In[ ]:


test_df


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


image_png = plt.imread('./fpmi_logo.png')


# In[ ]:


image_png


# In[ ]:


type(image_png)


# In[ ]:


image_png.shape


# In[ ]:


image_png[0].shape


# In[ ]:


image_png[0]


# In[ ]:


image_png.dtype


# In[ ]:


image_jpg = plt.imread('./dlschool_logo.jpg')


# In[ ]:


image_jpg.shape


# In[ ]:


plt.imshow(image_png);


# In[ ]:


plt.imshow(image_jpg);


# In[ ]:


train_df


# In[ ]:


test_df


# In[ ]:


train_df.values[0]


# In[ ]:


X_train = train_df.values[:, 1:]
y_train = train_df.values[:, 0]

X_test = test_df.values[:, :]  # удаляем столбец 'label'


# In[ ]:


print(X_train.shape, y_train.shape)


# In[ ]:


print(X_test.shape)


# In[ ]:


plt.imshow(X_train[0].reshape(28, 28), cmap='gray');


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

import torch


# In[ ]:


torch.__version__


# In[ ]:


X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.astype(np.int64))


# In[ ]:


print(X_train_tensor.shape, y_train_tensor.shape)


# In[ ]:


y_train_tensor.unique()


# In[ ]:


length = y_train_tensor.shape[0]
num_classes = 10  # количество классов, в нашем случае 10 типов одежды

# закодированные OneHot-ом метки классов
y_onehot = torch.FloatTensor(length, num_classes)

y_onehot.zero_()
y_onehot.scatter_(1, y_train_tensor.view(-1, 1), 1)

print(y_train_tensor)
print(y_onehot)


# In[ ]:


# N - размер батча (batch_size, нужно для метода оптимизации)
# D_in - размерность входа (количество признаков у объекта)
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
D_in, H, D_out = 784, 150, 10

# определим нейросеть:
net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.PReLU(),
    torch.nn.Linear(H, D_out),
    torch.nn.Softmax()
)


# In[ ]:


def generate_batches(X, y, batch_size=64):
    for i in range(0, X.shape[0], batch_size):
        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
        yield X_batch, y_batch


# In[ ]:


BATCH_SIZE = 64
NUM_EPOCHS = 200

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-4
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

for epoch_num  in range(NUM_EPOCHS):
    iter_num = 0
    running_loss = 0.0
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        # forward (подсчёт ответа с текущими весами)
        y_pred = net(X_batch)

        # вычисляем loss'ы
        loss = loss_fn(y_pred, y_batch)
        
        running_loss += loss.item()
        
        # выводем качество каждые 2000 батчей
            
        if iter_num % 2000 == 0:
            print('[{}, {}] current loss: {}'.format(epoch_num, iter_num + 1, running_loss / 2000))
            running_loss = 0.0
            
        # зануляем градиенты
        optimizer.zero_grad()

        # backward (подсчёт новых градиентов)
        loss.backward()

        # обновляем веса
        optimizer.step()
        
        iter_num += 1


# In[ ]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker','Bag', 'Ankle boot']

with torch.no_grad():
    for X_batch, y_batch in generate_batches(X_train_tensor, y_train_tensor, BATCH_SIZE):
        y_pred = net(X_batch)
        _, predicted = torch.max(y_pred, 1)
        c = (predicted == y_batch).squeeze()
        for i in range(len(y_pred)):
            label = y_batch[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


# In[ ]:


y_test_pred = net(torch.FloatTensor(X_test))


# In[ ]:


y_test_pred.shape


# In[ ]:


y_test_pred[:5]


# In[ ]:


_, predicted = torch.max(y_test_pred, 1)


predicted


# In[ ]:


answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])
answer_df.head()


# In[ ]:


answer_df['Id'] = answer_df.index


# In[ ]:


answer_df.head()


# In[ ]:


answer_df.tail()


# In[ ]:


answer_df.to_csv('./answer_my.csv', index=False)


# In[ ]:


<Ваш код здесь (может занимать много, очень много ячеек)> 


# In[ ]:


...


# In[ ]:


...

