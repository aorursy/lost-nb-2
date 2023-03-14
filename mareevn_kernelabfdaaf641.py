import pandas as pd

TRAIN_PATH = '../input/fashionmnist/fashion-mnist_train.csv'
TEST_PATH = '../input/fashionmnist/fashion-mnist_test.csv'

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

test_df

train_df.head()

test_df.head()

train_df.shape

test_df.shape

import numpy as np
import matplotlib.pyplot as plt

image_png = plt.imread('./fpmi_logo.png')

image_png

type(image_png)

image_png.shape

image_png[0].shape

image_png[0]

image_png.dtype

image_jpg = plt.imread('./dlschool_logo.jpg')

image_jpg.shape

plt.imshow(image_png);

plt.imshow(image_jpg);

train_df

test_df

train_df.values[0]

X_train = train_df.values[:, 1:]
y_train = train_df.values[:, 0]

X_test = test_df.values[:, 1:]  # удаляем столбец 'label'

print(X_train.shape, y_train.shape)

print(X_test.shape)

plt.imshow(X_train[0].reshape(28, 28), cmap='gray');

import matplotlib.pyplot as plt
import numpy as np

import torch

torch.__version__

X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.astype(np.int64))

print(X_train_tensor.shape, y_train_tensor.shape)

y_train_tensor.unique()

length = y_train_tensor.shape[0]
num_classes = 10  # количество классов, в нашем случае 10 типов одежды

# закодированные OneHot-ом метки классов
y_onehot = torch.FloatTensor(length, num_classes)

y_onehot.zero_()
y_onehot.scatter_(1, y_train_tensor.view(-1, 1), 1)

print(y_train_tensor)
print(y_onehot)

# N - размер батча (batch_size, нужно для метода оптимизации)
# D_in - размерность входа (количество признаков у объекта)
# H - размерность скрытых слоёв; 
# D_out - размерность выходного слоя (суть - количество классов)
D_in, H, D_out = 784, 300, 10

# определим нейросеть:
net = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),

    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
    #torch.nn.Softmax()
    torch.nn.LogSoftmax()
)

def generate_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        X_batch, y_batch = X[i:i+batch_size], y[i:i+batch_size]
        yield X_batch, y_batch

BATCH_SIZE = 16
NUM_EPOCHS = 50

loss_fn = torch.nn.CrossEntropyLoss(size_average=False)

learning_rate = 1e-5
#optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)#SGD(net.parameters(), lr=learning_rate)
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
            
        if iter_num % 100 == 99:
            print('[{}, {}] current loss: {}'.format(epoch_num, iter_num + 1, running_loss / 100))
            running_loss = 0.0
            
        # зануляем градиенты
        optimizer.zero_grad()

        # backward (подсчёт новых градиентов)
        loss.backward()

        # обновляем веса
        optimizer.step()
        
        iter_num += 1

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

total_a = 0.
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    total_a += (100 * class_correct[i] / class_total[i])
print(total_a/10)

#X_test.shape
X_test = torch.FloatTensor(X_test)

y_test_pred = net(X_test)

y_test_pred.shape

y_test_pred[:5]

_, predicted = torch.max(y_test_pred, 1)

predicted

answer_df = pd.DataFrame(data=predicted.numpy(), columns=['Category'])
answer_df.head()

answer_df['Id'] = answer_df.index

answer_df.head()

answer_df.tail()

answer_df.to_csv('./baseline.csv', index=False)
#'../input/fashionmnist/fashion-mnist_train.csv'

<Ваш код здесь (может занимать много, очень много ячеек)> 

...

...
