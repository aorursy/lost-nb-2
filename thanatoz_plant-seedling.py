# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
import cv2
import tensorflow as tf
import math
from tensorflow.python.framework import ops
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 33
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


root = '../input/train/Maize/3a6d4d007.png'
img = cv2.imread(root)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = cv2.resize(img,(128,128))
image_segmented = segment_plant(img)
image_sharpen = sharpen_image(image_segmented)
plt.imshow(image_sharpen)

    root = '../input/train'
    folders = os.listdir(root)
    X = []
    Y = []
    names={}
    ptr = 0
    for folder in  folders:
        names[ptr]=folder
        files = os.listdir(os.path.join(root,folder))
        for file in files:
            image_path = os.path.join(os.path.join(root,folder,file))
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            image_segmented = segment_plant(img)
            image_sharpen = sharpen_image(image_segmented)
            img = cv2.resize(image_sharpen,(128,128))
            img=img/255
            X.append(img)
            Y.append(ptr)
        ptr+=1

    X = np.array(X)
    Y = np.array(Y)
    names    

def display_dataset(X,Y, h=128, w=128, rows=5, cols=2, display_labels=True):
    f, ax = plt.subplots(cols, rows)
    for i in range(rows):
        for j in range(cols):
            index=np.random.randint(0,X.shape[0])
            ax[j,i].imshow(X[index].reshape(h,w,3), cmap='binary')
            ax[j,i].set_title(Y[index])
    plt.xticks()
    plt.show()

X = X.reshape(X.shape[0],-1)

X.shape

display_dataset(X, Y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, shuffle=True, test_size=0.1)

display_dataset(X_train, Y_train)

display_dataset(X_test, Y_test)

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')
    
    return X, Y

def initialize_parameters():
    
    tf.set_random_seed(1)                   # so that your "random" numbers match ours
        
    W1 = tf.get_variable("W1", [50, 49152], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [50, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [15, 50],initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [15, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [12, 15], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [12, 1], initializer=tf.zeros_initializer())
   
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters

def forward_propagation(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2) # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3) # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

# Compute_cost 

def compute_cost(Z3, Y):
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    print(logits, labels)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    
    return cost

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def predict(X, parameters, pred_val=1):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [49152, pred_val])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


def forward_propagation_for_predict(X, parameters):
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def model(train_X, train_Y, test_X, test_Y, learning_rate = 0.0001,
          num_epochs = 1000, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    """
    
    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)      # to keep consistent results
    seed = 3                   # to keep consistent results
    (n_x, m) = train_X.shape   # (n_x: input size, m : number of examples in the train set)
    n_y = train_Y.shape[0]     # n_y : output size
    costs = []                 # To keep track of the cost
    
    X, Y = create_placeholders(n_x, n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        
        sess.run(init)
        
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(train_X, train_Y, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                             feed_dict={X: minibatch_X, Y: minibatch_Y})

            epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: train_X, Y: train_Y}))
        print ("Test Accuracy:", accuracy.eval({X: test_X, Y: test_Y}))
        
        return parameters

Y_train = convert_to_one_hot(Y_train, 12)
Y_test = convert_to_one_hot(Y_test, 12)
print(Y_train.shape, Y_test.shape)

X_train=X_train.reshape(X_train.shape[0],-1).T
X_test=X_test.reshape(X_test.shape[0],-1).T
print(X_train.shape, X_test.shape)


# parameters = model(X_train, Y_train, X_test, Y_test, learning_rate=0.0001, num_epochs=260)

parameters1 = model(X_train, Y_train, X_test, Y_test, learning_rate=0.001, num_epochs=100)

root = '../input/train/Maize/3a6d4d007.png'
imag = cv2.imread(root)
imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
imag = cv2.resize(imag,(128,128))
image_segmented = segment_plant(imag)
image_sharpen = sharpen_image(image_segmented)
imag = cv2.resize(image_sharpen,(128,128))
imag = imag/255

imb = imag.reshape(1, 128*128*3).T
my_image_prediction = predict(imb, parameters1)
plt.imshow(imb.reshape(128,128,3))
print("Your algorithm predicts: y = " + str(names[int(np.squeeze(my_image_prediction))]))

root = '../input/test/'
files = os.listdir(root)
x=[]
y=[]
for file in files:
    y.append(file)
    imag = cv2.imread(os.path.join(root,file))
    imag = cv2.cvtColor(imag, cv2.COLOR_RGB2BGR)
    imag = cv2.resize(imag,(128,128))
    image_segmented = segment_plant(imag)
    imag = sharpen_image(image_segmented)
#     imag = cv2.resize(image_sharpen,(128,128))
    imag = imag/255
#     print(ctr)
    imb = imag.reshape(-1)
    x.append(imb)
x=np.array(x)
y=np.array(y)
x = x.reshape(x.shape[0],-1).T
print(x.shape, y.shape)

preda = predict(x, parameters1, x.shape[1])

import pickle
pickle.dump(parameters1, open('parameters_pickle.p','wb'))
pickle.dump(names, open('parameters_pickle.p','wb'))

put = [names[i] for i in preda]
put

res = pd.DataFrame({'file':y, 'species':put })
res.head()

res.to_csv('submission.csv', index=False)


