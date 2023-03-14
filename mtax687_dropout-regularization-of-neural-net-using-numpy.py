#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import sys
import datetime
## pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## constants
TRAIN_DIR = "./input/train/"
TEST_DIR = "./input/test/"
TRAIN_SIZE = 25000
TEST_SIZE = 12500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128

LEARNING_RATE = 0.0001
MINIBATCH_SIZE = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
OUTPUT_SIZE = 2


# In[3]:


## data utility functions
def split_data(two_dims_datas, split_ratio=DEV_RATIO):
    left_count = int(two_dims_datas.shape[1] * split_ratio)
    left_datas = two_dims_datas[:, :left_count]
    right_datas = two_dims_datas[:, left_count:]
    print("input datas shape: {}, left datas shape:{},     right datas shape: {}".format(two_dims_datas.shape, left_datas.shape, right_datas.shape))
    return left_datas, right_datas


# In[4]:


def load_data(dirname=TRAIN_DIR, file_count=1000, shuffle=True):
    all_filenames = os.listdir(dirname)
    random.shuffle(all_filenames)
    filenames = all_filenames[:file_count]
    
    ## images
    images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    for i in range(file_count):
        imgnd_origin = cv2.imread(dirname+filenames[i])
        imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        imgnd_flatten = imgnd_resized.reshape(1,-1)
        images[i] = imgnd_flatten
    
    ## labels from filenames
    labels_list = ["dog" in filename for filename in filenames]
    labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)
    
    ## shuffle
    if shuffle:
        permutation = list(np.random.permutation(labels.shape[0]))
        labels = labels[permutation, :]
        images = images[permutation, :]

    ## normalization
    images = images/255.0
    
    return images.T, labels.T


# In[5]:


images, labels = load_data(file_count=200)
dev_images, train_images = split_data(images)
dev_labels, train_labels = split_data(labels)


# In[6]:


n_x = train_images.shape[0] # size of input layer
n_h = 4          # hard code the hidden layer size to be 4
n_y = train_labels.shape[0] # size of output layer

layer_sizes = (n_x, n_h, n_y)
print(layer_sizes)


# In[7]:


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[8]:


## SIGMOID

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    s = 1.0/(1.0 + np.exp(-1.0 * z))
    
    return s


# In[9]:


# forward_propagation
# forward_propagation
def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    #print(W1.shape, X.shape)
    #print(np.matmul(W1, X).shape, b1.shape)
    Z1 = np.add(np.matmul(W1, X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.matmul(W2, A1), b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

def forward_propagation_with_dropout(X, parameters, keep_prob = 0.5):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1)
    keep_prob -- probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.add(np.matmul(W1, X), b1)
    A1 = np.tanh(Z1)
    ## Dropout
    D1 = np.random.rand(A1.shape[0], A1.shape[1])                               # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D1 = D1 < keep_prob                                         # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = np.multiply(A1, D1)                                         # Step 3: shut down some neurons of A1
    A1 = A1/keep_prob                                         # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.add(np.matmul(W2, A1), b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "D1": D1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# In[10]:


# compute_cost

def compute_cost(A2, Y, parameters):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    
    Returns:
    cost -- cross-entropy cost given equation (13)
    """
    
    m = Y.shape[1] # number of example

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    #print(logprobs.shape)
    cost = (-1.0/m) * np.sum(logprobs)
    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect. 
                                # E.g., turns [[17]] into 17 
    assert(isinstance(cost, float))
    
    return cost


# In[11]:


# backward_propagation

def backward_propagation_with_dropout(parameters, cache, X, Y, keep_prob):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    D1 = cache["D1"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. 
    dZ2 = A2 - Y
    dW2 = (1.0/m) * np.matmul(dZ2, np.transpose(A1))
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    ## Dropout
    dA1 = np.multiply(D1, dA1)        # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA1 = dA1/keep_prob              # Step 2: Scale the value of neurons that haven't been shut down
    
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1.0/m) * np.matmul(dZ1, np.transpose(X))
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# In[12]:


# update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[13]:


# nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate=1.2, keep_prob=0.5, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    costs = []
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation_with_dropout(X, parameters, keep_prob = keep_prob)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
        costs.append(cost)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation_with_dropout(parameters, cache, X, Y, keep_prob)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        
        # cache all about model
        trained_model = {
            "layer_sizes": (n_x, n_h, n_y),
            "learning_rate": learning_rate,
            "costs": costs,
            "parameters": parameters
        }

        # Print the cost every 100 iterations
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return trained_model


# In[14]:


# predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions


# In[15]:


def plot_costs(trained_model):
    # Plot learning curve (with costs)
    costs = np.squeeze(trained_model["costs"])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate ={}\n, layer_sizes={}\n, accuracy:{}, m:{}\n".format(
        trained_model["learning_rate"], trained_model["layer_sizes"],
        trained_model.get("accuracy"), trained_model.get("m"), trained_model.get("keep_probs")
    ))
    #plt.show()
    


# In[16]:


# Example of a picture that was wrongly classified.
trained_model = nn_model(train_images, train_labels, n_h, num_iterations = 50)

index = 11
plt.imshow(dev_images[:,index].reshape((IMAGE_HEIGHT, IMAGE_HEIGHT, 3)))
predictions = predict(trained_model["parameters"], dev_images[:,index:index+1])
print ("y = {}, you predicted that it is a {}".format(dev_labels[0,index],int(predictions)))


# In[17]:


# This may take about 1 minutes to run

images, labels = load_data(file_count=10000)
dev_images, train_images = split_data(images)
dev_labels, train_labels = split_data(labels)


# In[18]:


# This may take about 2 minutes to run
m = 9800
keep_probs = [1.0, 0.7, 0.5, 0.3]
num_iterations=1001
learning_rate=0.05
trained_models = []

print(datetime.datetime.now())
for prob in keep_probs:
    trained_model = nn_model(train_images, train_labels, 20,
                             num_iterations = num_iterations, learning_rate=learning_rate,
                             keep_prob=prob, print_cost=True
                            )
    predictions = predict(trained_model["parameters"], dev_images)
    accuracy = float((np.dot(dev_labels,predictions.T) + np.dot(1-dev_labels,1-predictions.T))/float(dev_labels.size)*100)
    print ("Accuracy for {} keep_prob : {} %".format(prob, accuracy))
    ## cache trained_model
    trained_model["accuracy"] = accuracy
    trained_model["m"] = m
    trained_model["num_iterations"] = num_iterations
    trained_model["keep_probs"] = prob
    trained_models.append(trained_model)
    ## plot costs
    plt.figure(num=None, figsize=(15, 6), dpi=50, facecolor='w', edgecolor='k')
    plot_costs(trained_model)

    print(datetime.datetime.now())


# In[19]:


import pandas as pd
df = pd.DataFrame(trained_models)


# In[20]:


df.head(10)


# In[21]:


len(trained_models)


# In[22]:


plt.figure(num=None, figsize=(20, 20), dpi=200, facecolor='w', edgecolor='k')
for i in range(len(trained_models)):
    plt.subplot(921+i)
    plot_costs(trained_models[i])


# In[23]:




