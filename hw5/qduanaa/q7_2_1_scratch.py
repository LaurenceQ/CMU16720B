import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
dtype = torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = IMAGENET_MEAN, std = IMAGENET_STD)
])
batch_size = 64
trainset = torchvision.datasets.ImageFolder('../data/oxford-flowers17/train', transform=transform)
train_loader = DataLoader(trainset, batch_size = batch_size, shuffle = True)
validset = torchvision.datasets.ImageFolder('../data/oxford-flowers17/val', transform=transform)
valid_loader = DataLoader(validset, batch_size = batch_size, shuffle = False)
testset = torchvision.datasets.ImageFolder('../data/oxford-flowers17/test', transform=transform)
test_loader = DataLoader(testset, batch_size = batch_size, shuffle = False)
learning_rate = 0.0005
max_iters = 200

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding = 'same')
        self.conv2 = nn.Conv2d(6, 12, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(12, 6, 3, stride = 2, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(6 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 17)

    def forward(self, x):
        output = F.relu(self.conv1(x))
        #output = self.pool(output)
        output = F.relu(self.conv2(output))
        output = self.pool(output)
        output = F.relu(self.conv3(output))
        output = self.pool2(output)
        output = torch.flatten(output, 1)
        output = torch.tanh(self.fc1(output))
        output = F.softmax(self.fc2(output), dim = -1)
        return output

def train_accuracy_loss(model, loader, optimizer):
    total_loss = 0
    total_acc = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        probs = model(xb)
        loss = criterion(probs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, pred = torch.max(probs.detach(), 1)
        total_acc += (pred == yb).sum().item() 
    total_acc /= len(loader.dataset)
    return total_acc, total_loss


def get_accuracy_loss(model, loader):
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for data in loader:
            xb, yb = data
            #xb, yb = xb.cuda(), yb.cuda()
            probs = model(xb)
            loss = criterion(probs, yb)
            total_loss += loss.item()
            _, pred = torch.max(probs.detach(), 1)
            total_acc += (pred == yb).sum().item() 
    total_acc = total_acc / len(loader.dataset)
    return total_acc, total_loss

def print_acc_loss(train_acc, train_loss, val_acc, val_loss):
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

model = Net()
train_loss = []
train_acc = []
val_loss = []
val_acc = []
criterion = nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for itr in range(max_iters):
    total_acc, total_loss = train_accuracy_loss(model, train_loader, model_optimizer)
    train_acc.append(total_acc)
    train_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    total_acc, total_loss = get_accuracy_loss(model, valid_loader)
    val_acc.append(total_acc )
    val_loss.append(total_loss)

print_acc_loss(train_acc, train_loss, val_acc, val_loss)

PATH = './flower17_cifar_net.pth'
torch.save(model.state_dict(), PATH)
total_acc, total_loss = get_accuracy_loss(model, test_loader)
print("test acc:", total_acc)
print("test loss:", total_loss)

