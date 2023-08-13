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
model = torchvision.models.squeezenet1_1(weights = 'IMAGENET1K_V1')
num_classes = 17
final_conv = nn.Conv2d(512, num_classes, kernel_size=1)
model.classifier = nn.Sequential(
    nn.Dropout(), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
)
model.type(torch.FloatTensor)
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True
train_loss = []
train_acc = []
val_loss = []
val_acc = []
criterion = nn.CrossEntropyLoss()
model_optimizer = torch.optim.Adam(model.classifier.parameters(), lr = 1e-3)
#optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
max_iters = 10
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
train_acc = []
train_loss = []
val_acc = []
val_loss = []
for param in model.parameters():
    param.requires_grad = True

model_optimizer = torch.optim.Adam(model.parameters(), lr = 1e-5)

max_iters2 = 20
for itr in range(max_iters2):
    total_acc, total_loss = train_accuracy_loss(model, train_loader, model_optimizer)
    train_acc.append(total_acc)
    train_loss.append(total_loss)
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
    
    total_acc, total_loss = get_accuracy_loss(model, valid_loader)
    val_acc.append(total_acc )
    val_loss.append(total_loss)

print_acc_loss(train_acc, train_loss, val_acc, val_loss)


PATH = './flower17_pretrained_cifar_net.pth'
torch.save(model.state_dict(), PATH)


total_acc, total_loss = get_accuracy_loss(model, test_loader)
print("test acc:", total_acc)
print("test loss:", total_loss)

