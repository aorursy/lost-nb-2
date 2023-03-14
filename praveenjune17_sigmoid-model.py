#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
path = "../input/"
test = pd.read_json("../input/test.json")
train = pd.read_json("../input/train.json")


# In[2]:


#train[train['inc_angle']=='na']


# In[3]:





# In[3]:





# In[3]:





# In[3]:


#test.head()


# In[4]:





# In[4]:





# In[4]:





# In[4]:





# In[4]:




def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    
    s = 1/(1+np.exp(-z))
    
    
    return s


# In[5]:





# In[5]:





# In[5]:





# In[5]:





# In[5]:




def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    
    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    
    w = np.zeros(dim).reshape(dim,1)
    b = 0
    

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    
    return w, b


# In[6]:





# In[6]:





# In[6]:





# In[6]:





# In[6]:




def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    J=−1m∑mi=1y(i)log(a(i))+(1−y(i))log(1−a(i))J=−1m∑i=1my(i)log⁡(a(i))+(1−y(i))log⁡(1−a(i))
    
    
    """
    
    
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    mat1=(Y*np.log(A))
    mat2=((1-Y)*np.log(1-A))
    
    
    cost = -1/m*np.sum(mat1+mat2)
    
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    
    dw = 1/m*(np.dot(X, (A-Y).T))
    db = 1/m*np.sum((A-Y))
    

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


# In[7]:





# In[7]:





# In[7]:





# In[7]:





# In[7]:




def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        
        
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule 
        
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


# In[8]:





# In[8]:





# In[8]:





# In[8]:





# In[8]:




def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    
    A = sigmoid(np.dot(w.T, X) + b)
    
    
    Y_prediction = np.array(((A > 0.5).squeeze()*1).reshape(1,m))
    
    
    assert(Y_prediction.shape == (1, m))
    
    return (Y_prediction,A)


# In[9]:





# In[9]:





# In[9]:





# In[9]:





# In[9]:





# In[9]:





# In[9]:




def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test, A_test = predict(w, b, X_test)
    Y_prediction_train, A_train = predict(w, b, X_train)

    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "train_with_prob" : A_train,
         "test_with_prob" : A_test,
         "num_iterations": num_iterations}
    #print A
    return d


# In[10]:





# In[10]:





# In[10]:





# In[10]:





# In[10]:





# In[10]:





# In[10]:





# In[10]:





# In[10]:


def train_test_split_fun(array_in, array_out, split_perc=0.25):
    from sklearn.model_selection import train_test_split
    X_train, X_val_test, y_train, y_val_test = train_test_split(array_in.T, array_out.T,
                                                    stratify=array_out.T, 
                                                     test_size=split_perc)
    dataset = (X_train.T, X_val_test.T, y_train.T, y_val_test.T)
    return dataset


# In[11]:





# In[11]:


#len(train[train['inc_angle']!='na'])


# In[12]:





# In[12]:





# In[12]:


def JSON_to_array(split_perc = 0.25, file='train.json'):

#input the json file to the function. It converts the file to df and returns the numpy array 
#The array of columns band_1 and band_2 are returned as numpy array

    train_set=pd.read_json(path+file)
    inc_set = train_set[train_set['inc_angle']!='na']
    
    band_1=[np.array(i) for i in train_set['band_1']]
    band_2=[np.array(i) for i in train_set['band_2']]
    inc_ang = [np.array(i) for i in inc_set['inc_angle']]

    ice_berg=[np.array(i) for i in train_set['is_iceberg']]
    ice_berg = np.array(ice_berg).reshape(1, 1604)
    inc_set_ice_berg = [np.array(i) for i in inc_set['is_iceberg']]
    inc_set_ice_berg = np.array(inc_set_ice_berg).reshape(1, 1471)

    max_band_1 = np.max(np.array(np.abs(band_1)))
    max_band_2 = np.max(np.array(np.abs(band_2)))
    max_inc_ang = np.max(np.array(np.abs(inc_ang)))
    
    band_2 = np.array(band_2).T/max_band_2
    band_1 = np.array(band_1).T/max_band_1
    inc_ang = np.array(inc_ang).T/max_inc_ang
    inc_ang = inc_ang.reshape(1, 1471)
    
    #print(inc_ang.shape, inc_set_ice_berg.shape)
    #print(band_2.shape, ice_berg.shape)
    
    band_1_array = train_test_split_fun(band_1, ice_berg, split_perc=split_perc)    
    band_2_array = train_test_split_fun(band_2, ice_berg, split_perc=split_perc)    
    inc_ang_array = train_test_split_fun(inc_ang, inc_set_ice_berg, split_perc=split_perc)    
    
    return(band_1_array, band_2_array, inc_ang_array)


# In[13]:





# In[13]:





# In[13]:


#train_set=pd.read_json(path+'train.json')
#inc_ang = [np.array(i) for i in train_set['inc_angle']]
#max_inc_ang = np.max(np.array(np.abs(inc_ang)))


# In[14]:


(band_1_array, band_2_array, inc_ang_array) = JSON_to_array(file='train.json')


# In[15]:





# In[15]:





# In[15]:





# In[15]:





# In[15]:





# In[15]:





# In[15]:





# In[15]:


#X_train_band_1, X_val_test_band_1, y_train_band_1, y_val_test_band_1 = band_1_array


# In[16]:


#X_train_band_1.shape


# In[17]:





# In[17]:


#X_train_inc, X_val_test_inc, y_train_inc, y_val_test_inc = inc_ang_array


# In[18]:





# In[18]:





# In[18]:


#X_train_inc.shape


# In[19]:





# In[19]:


#X_val_test_inc.shape


# In[20]:





# In[20]:


#band_2 :- iteration = 15000 , rate = 0.005
# train accuracy: 86.78304239401496 %
#test accuracy: 64.33915211970074 %
#with learning_Rate = 0.001
#train accuracy: 76.14297589359933 %
#test accuracy: 64.83790523690773 %


# In[21]:


X_train_band_1, X_val_test_band_1, y_train_band_1, y_val_test_band_1 = band_1_array
d_band_1 = model(X_train_band_1, y_train_band_1, X_val_test_band_1, y_val_test_band_1, num_iterations = 15000, learning_rate = 0.005, print_cost = True)


# In[22]:





# In[22]:





# In[22]:


X_train_band_2, X_val_test_band_2, y_train_band_2, y_val_test_band_2 = band_2_array
d_band_2 = model(X_train_band_2, y_train_band_2, X_val_test_band_2, y_val_test_band_2, num_iterations = 15000, learning_rate = 0.001, print_cost = True)


# In[23]:





# In[23]:


X_train_inc, X_val_test_inc, y_train_inc, y_val_test_inc = inc_ang_array
d_inc_ang = model(X_train_inc, y_train_inc, X_val_test_inc, y_val_test_inc, num_iterations = 15000, learning_rate = 0.001, print_cost = True)


# In[24]:





# In[24]:


test_band_1=[np.array(i) for i in test['band_1']]
test_band_2=[np.array(i) for i in test['band_2']]
test_inc_ang=[np.array(i) for i in test['inc_angle']]

max_band_1 = np.max(np.array(np.abs(test_band_1)))
max_band_2 = np.max(np.array(np.abs(test_band_2)))
max_inc_ang = np.max(np.array(np.abs(test_inc_ang)))

band_2 = np.array(test_band_2).T/max_band_2
band_1 = np.array(test_band_1).T/max_band_1
inc_ang = (np.array(test_inc_ang).T/max_inc_ang).reshape(1, 8424)

binary_result_band_1, band_1_Y_prediction_test = predict(d_band_1['w'], d_band_1['b'], band_1)
y_pred_band_1 = pd.DataFrame(band_1_Y_prediction_test.T,columns=['is_iceberg'])

binary_result_band_2, band_2_Y_prediction_test = predict(d_band_2['w'], d_band_2['b'], band_2)
y_pred_band_2 = pd.DataFrame(band_2_Y_prediction_test.T,columns=['is_iceberg'])

binary_result_band_3, inc_ang_Y_prediction_test = predict(d_inc_ang['w'], d_inc_ang['b'], inc_ang)
y_pred_inc_ang = pd.DataFrame(inc_ang_Y_prediction_test.T,columns=['is_iceberg'])


#y_pred_band_1['id'] = test['id']
#cols = y_pred_band_1.columns.tolist()
#cols =cols[-1:] + cols[:-1]
#y_pred_band_1 = y_pred_band_1[cols]
#y_pred_band_1.to_csv('submission_2.csv', index=False)


# In[25]:





# In[25]:





# In[25]:


y_pred = pd.DataFrame()
y_pred['id'] = test['id']
#y_pred['is_iceberg'] = None
y_pred['is_iceberg'] = ((y_pred_band_1['is_iceberg']) + (y_pred_band_2['is_iceberg']) +(y_pred_inc_ang['is_iceberg']))/3
y_pred.to_csv('submission_3.csv', index=False)


# In[26]:





# In[26]:





# In[26]:


#y_pred['is_iceberg'] = ((y_pred_band_1['is_iceberg']) + (y_pred_band_2['is_iceberg']) +(y_pred_inc_ang['is_iceberg']))/3


# In[27]:





# In[27]:





# In[27]:





# In[27]:


y_pred.head()


# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:





# In[28]:


#y_pred.to_csv('submission_3.csv', index=False)


# In[29]:


y_pred


# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:





# In[30]:


y_pred


# In[31]:





# In[31]:


len(test)


# In[32]:


test['prediction'] = None


# In[33]:


test['prediction'] = 


# In[34]:





# In[34]:





# In[34]:





# In[34]:





# In[34]:





# In[34]:





# In[34]:


d['w'].shape


# In[35]:





# In[35]:





# In[35]:





# In[35]:


band1_x_train, band1_x_train 


# In[36]:





# In[36]:





# In[36]:




