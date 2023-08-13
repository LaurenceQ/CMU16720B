import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),
])
batch_size = 64
trainset = torchvision.datasets.EMNIST('../data', split = 'balanced', train = True, download = True, transform=transform)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
trainset.train_data.to(torch.device("cuda:0"))
validset = torchvision.datasets.EMNIST('../data', split = 'balanced', train = False, download = True, transform=transform)
valid_loader = DataLoader(validset, batch_size = batch_size, shuffle = False)
validset.train_data.to(torch.device("cuda:0"))

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, padding = 'same')
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(24, 48, 3, padding = 'same')
        self.fc1 = nn.Linear(48 * 49, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 47)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        output = self.pool(output)
        output = F.relu(self.conv3(output))
        output = self.pool(output)
        output = torch.flatten(output, 1)
        output = torch.tanh(self.fc1(output))
        output = torch.tanh(self.fc2(output))
        output = torch.softmax(self.fc3(output), dim = -1)
        return output

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        
net = Net()
net.apply(init_weights)
net.to(torch.device("cuda:0"))
train_loss = []
train_acc = []
val_loss = []
val_acc = []
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
max_iters = 16
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for i, data in enumerate(train_loader, 0):
        xb, yb = data
        xb, yb = xb.cuda(), yb.cuda()
        optimizer.zero_grad()
        probs = net(xb)
        loss = criterion(probs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(probs.detach(), 1)
        total_acc += (pred == yb).sum().item() 
        
    #print(len(train_loader.dataset), total_acc)
    total_acc /= len(train_loader.dataset)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    train_acc.append(total_acc)
    train_loss.append(total_loss)
    total_loss = total_acc = 0
    with torch.no_grad():
        for i, data in enumerate(valid_loader, 0):
            xb, yb = data
            xb, yb = xb.cuda(), yb.cuda()
            probs = net(xb)
            loss = criterion(probs, yb)
            total_loss += loss.item()
            _, pred = torch.max(probs.detach(), 1)
            total_acc += (pred == yb).sum().item() / xb.shape[0]
    val_acc.append(total_acc / len(valid_loader))
    val_loss.append(total_loss)

PATH = './EMINST_cnn_cifar_net.pth'
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