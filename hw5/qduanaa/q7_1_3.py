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
        self.conv1 = nn.Conv2d(1, 3, 3, padding = 'same')
        #self.conv2 = nn.Conv2d(3, 3, 3, padding = 'same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(3, 3, 3, padding = 'same')
        self.conv4 = nn.Conv2d(3, 3, 3, padding = 'same')

        #self.conv5 = nn.Conv2d(3, 3, 3, padding = 'same')
        #self.conv6 = nn.Conv2d(3, 3, 3, padding = 'same')

        #self.conv7 = nn.Conv2d(3, 3, 3, padding = 'same')
        #self.conv8 = nn.Conv2d(3, 3, 3, padding = 'same')
        self.fc1 = nn.Linear(3 * 64, 64)
        self.fc2 = nn.Linear(64, 36)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        #output = F.relu(self.conv2(output))
        output = self.pool(output)
        #temp = output
        output = F.relu(self.conv3(output))
        #output = F.relu(self.conv4(output))
        #temp = output
        #output = F.relu(self.conv5(output))
        #output = F.relu(self.conv6(output) + temp)
        #temp = output
        #output = F.relu(self.conv7(output))
        #output = F.relu(self.conv8(output) + temp)
        output = self.pool(output)
        
        output = torch.flatten(output, 1)
        output = torch.tanh(self.fc1(output))
        output = torch.softmax(self.fc2(output), dim = -1)
        return output

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
train_x = train_x.reshape(-1, 1, 32, 32)
valid_x = valid_x.reshape(-1, 1, 32, 32)
train_dataset = TensorDataset(torch.from_numpy(train_x.astype(np.float32)), torch.from_numpy(train_y.astype(np.float32)))
valid_dataset = TensorDataset(torch.from_numpy(valid_x.astype(np.float32)), torch.from_numpy(valid_y.astype(np.float32)))
max_iters = 50
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

PATH = './cnn36_cifar_net.pth'
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