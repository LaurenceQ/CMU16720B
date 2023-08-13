import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import scipy.io
import matplotlib.pyplot as plt

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(1024, 64)
        self.output = nn.Linear(64, 36)

    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = F.softmax(self.output(x), dim = -1)
        return x

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        
net = Net()
net.apply(init_weights)
print(net)

params = list(net.parameters())

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
train_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
valid_dataset = TensorDataset(torch.from_numpy(valid_x.astype(np.float32)), torch.from_numpy(valid_y.astype(np.float32)))
max_iters = 200
batch_size = 32
learning_rate = 0.001
train_loader = DataLoader(train_dataset, batch_size, True)
valid_loader = DataLoader(valid_dataset, batch_size, True)
print(len(train_loader))
print(len(valid_loader))
train_loss = []
train_acc = []
val_loss = []
val_acc = []
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for i, data in enumerate(train_loader, 0):
        xb, yb = data
        optimizer.zero_grad()
        probs = net(xb)
        loss = criterion(probs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(probs.detach(), 1)
        _, label = torch.max(yb.detach(), 1)
        total_acc += (pred == label).sum().item() 
        #print(xb.shape[0], total_acc)
        #exit(0)
        
    total_acc /= len(train_loader.dataset)
    train_acc.append(total_acc)
    train_loss.append(total_loss)
    total_loss = total_acc = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            xb, yb = data
            probs = net(xb)
            loss = criterion(probs, yb)
            total_loss += loss.item()
            _, pred = torch.max(probs.detach(), 1)
            _, label = torch.max(yb.detach(), 1)
            total_acc += (pred == label).sum().item() / xb.shape[0]
    val_acc.append(total_acc / len(valid_loader))
    val_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)
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