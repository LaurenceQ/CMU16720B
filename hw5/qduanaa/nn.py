import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None
    b = np.zeros(out_size)
    rang = np.sqrt(6 / (in_size + out_size))
    W = np.random.uniform(-rang, rang, (in_size, out_size))
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    res = np.ones(x.shape) / (np.ones(x.shape) + np.exp(-x))
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    
    pre_act = np.matmul(X, W) + b
    post_act = activation(pre_act)
    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None
    x = x - np.max(x, axis = 1, keepdims = True)
    x = np.exp(x)
    res = x / np.sum(x, axis = 1, keepdims = True)
    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
#from sklearn.metrics import log_loss
def compute_loss_and_acc(y, probs):
    loss, acc = None, None
    acc = np.argmax(probs, axis = 1)
    N = y.shape[0]
    acc = np.sum(y[np.arange(N), acc]) / N
    loss = - np.log(probs)
    loss = loss[np.arange(N), np.argmax(y, axis = 1)]
    #print(loss.shape)
    loss = np.sum(loss)
    #loss == log_loss(y, probs)
    #exit(0)
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    delta = delta * activation_deriv(post_act)
    grad_W = np.matmul(X[:,:,None], delta[:,None,:]).sum(0)
    grad_X = np.matmul(W, delta.T).T
    grad_b = delta.sum(0)
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0]
    ind = np.arange(N)
    np.random.shuffle(ind)
    ind = np.array_split(ind, N // batch_size)
    for i in range(len(ind)):
        batches.append((x[ind[i]], y[ind[i]]))
    return batches
