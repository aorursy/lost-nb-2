from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy as np
import time

trX = np.linspace(-1,1,100)
trY = 2 * trX + np.random.randn(*trX.shape)*0.33

def model(X,w):
    return X*w
w = theano.shared(np.asarray(0.,dtype=theano.config.floatX))
y = model(X,w)

cost = T.mean(T.sqrt
