import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
print(train_x.shape)
print(train_y.shape)
max_iters = 50
# pick a batch size, learning rate
batch_size = 20
learning_rate = 0.01
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')
W_layer1 = np.zeros(params['Wlayer1'].shape)
W_layer1[:,:] = params['Wlayer1']
# with default settings, you should get loss < 150 and accuracy > 80%
train_loss = []
train_acc = []
val_loss = []
val_acc = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        # forward
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss / xb.shape[0]
        total_acc += acc
        # loss
        # be sure to add loss and accuracy to epoch totals 
        delta1 = probs
        delta1[np.arange(probs.shape[0]),yb.argmax(axis = 1)] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        # backward
        for name in ['layer1', 'output']:
            W = params['W'+name]
            b = params['b'+name]
            grad_W = params['grad_W'+name]
            grad_b = params['grad_b'+name]
            W -= learning_rate * grad_W
            b -= learning_rate * grad_b
            params['W'+name] = W
            params['b'+name] = b
    total_acc /= len(batches)
    train_acc.append(total_acc)
    train_loss.append(total_loss / len(batches))
    h1 = forward(valid_x,params,'layer1')
    probs = forward(h1,params,'output',softmax)
    loss, acc = compute_loss_and_acc(valid_y, probs)
    val_acc.append(acc)
    val_loss.append(loss / valid_x.shape[0])
    # training loop can be exactly the same as q2!
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

#plot the acc and loss
'''
plt.figure(figsize=(10,5))
plt.title("Training and Validation Acc")
plt.plot(val_acc,label="val")
plt.plot(train_acc,label="train")
plt.xlabel("iterations")
plt.ylabel("Acc")
plt.legend()
plt.show()
plt.figure(figsize=(10,5))
plt.title("Training and Validation Averaged Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
'''
# run on validation set and report accuracy! should be above 75%
valid_acc = 0
h1 = forward(valid_x,params,'layer1')
probs = forward(h1,params,'output',softmax)
# forward
loss, acc = compute_loss_and_acc(valid_y, probs)
valid_acc = acc
# loss
# be sure to add loss and accuracy to epoch totals 
print('Validation accuracy: ',valid_acc)
print(loss)
control = False
if control: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T, cmap = 'gray')
        plt.show()
        print(crop[0].mean())
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
#import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
W_layer1 = W_layer1.reshape((32, 32, hidden_size))
fig = plt.figure(1, (8.0, 8.0))
grid = ImageGrid(fig, 111, nrows_ncols = (8, 8), axes_pad = 0.1)

for i in range(hidden_size):
    grid[i].imshow(W_layer1[:,:,i])
plt.show()
W_layer1 = W_layer1.reshape((1024, hidden_size))
W_layer1[:,:] = params['Wlayer1']
W_layer1 = W_layer1.reshape((32, 32, hidden_size))
fig = plt.figure(2, (8.0, 8.0))
grid = ImageGrid(fig, 111, nrows_ncols = (8, 8), axes_pad = 0.1)

for i in range(hidden_size):
    grid[i].imshow(W_layer1[:,:,i])
plt.show()
# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for i in range(valid_y.shape[0]):
    confusion_matrix[valid_y[i].argmax()][probs[i].argmax()] += 1
import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()