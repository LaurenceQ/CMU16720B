import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

# initialize layers here
params = Counter()
initialize_weights(1024,32,params,'layer1')
initialize_weights(32, 32, params, 'hidden')
initialize_weights(32, 32, params, 'hidden2')
initialize_weights(32,1024,params,'output')


names = ['layer1', 'hidden', 'hidden2', 'output']
# should look like your previous training loops
losses = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1 = xb
        for i in range(len(names) - 1):
            h1 = forward(h1, params, names[i], relu)
        res = forward(h1, params, names[-1], sigmoid)
        loss = np.sum((res - xb) ** 2)
        total_loss += loss
        delta1 = 2 * (res - xb)
        delta2 = backwards(delta1,params,'output',sigmoid_deriv)
        delta3 = backwards(delta2, params, names[2], relu_deriv)
        delta4 = backwards(delta3, params, names[1], relu_deriv)
        backwards(delta4, params, names[0], relu_deriv)
        for name in names:
            W = params['W'+name]
            b = params['b'+name]
            Moment = params['m_'+name]
            grad_W = params['grad_W'+name]
            grad_b = params['grad_b'+name]
            Moment = 0.9 * Moment - learning_rate * grad_W
            W += Moment
            b -= learning_rate * grad_b
            params['W'+name] = W
            params['b'+name] = b
            params['m_'+name] = Moment
 
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
    losses.append(total_loss)

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
'''plt.figure(figsize=(10,5))
plt.title("Training Loss")
plt.plot(losses,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
'''
xb = valid_x
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'hidden',relu)
h3 = forward(h2,params,'hidden2',relu)
out = forward(h3,params,'output',sigmoid)
valid_y = valid_data['valid_labels'].argmax(1)
cnt = np.zeros(5)
for i in range(len(valid_x)):
    if valid_y[i] >= 5:continue
    if cnt[valid_y[i]] == 2: continue
    cnt[valid_y[i]] += 1
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


#from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr
total_psnr = 0
for i in range(len(valid_x)):
    total_psnr += psnr(valid_x[i], out[i])
print("total psnr:",total_psnr)
print("average psnr:", total_psnr / len(valid_x))
# evaluate PSNR
# Q5.3.2

