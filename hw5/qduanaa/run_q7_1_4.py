import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

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


def cmp(v):
    return v[1]
correct_seneteces = [
    [
        "TODOLIST",
        "1MAKEATODOLIST",
        "2CHECKOFFTHEFIRST",
        "THINGONTODOLIST",
        "3REALIZEYOUHAVEALREADY",
        "COMPLETED2THINGS",
        "4REWARDYOURSELFWITH",
        "ANAP"
    ]
    ,
    [
        "ABCDEFG",
        "HIJKLMN",
        "OPQRSTU",
        "VWXYZ",
        "1234567890"
    ]
    ,
    [
        "HAIKUSAREEASY",
        "BUTSOMETIMESTHEYDONTMAKESENSE",
        "REFRIGERATOR"
    ]
    ,
    [
        "DEEPLEARNING",
        "DEEPERLEARNING",
        "DEEPESTLEARNING"
    ]
]
net = Net()
PATH = './EMINST_cnn_cifar_net.pth'
net.load_state_dict(torch.load(PATH))
for param in net.parameters():
    param.requires_grad = False
cnt = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    bw = 1 - bw
    bboxes = sorted(bboxes, key = cmp)
    plt.imshow(bw, 'gray')
    max_y_len = 0
    max_x_len = 0
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        max_y_len = max(max_y_len, maxr - minr)
        max_x_len = max(max_x_len, maxc - minc)
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    Ys = []
    K = 0
    iter_num = 800
    mark = np.zeros(len(bboxes))
    group = []
    for k in range(1, 9):
        max_match = 0
        cur_Y = None
        for iter in range(iter_num):
            temp_Y = np.random.uniform(0, bw.shape[0], (1,))
            temp_match = 0
            mean_Y = 0
            for i in range(len(bboxes)):
                if mark[i] != 0 : continue
                y1, x1, y2, x2 = bboxes[i]
                lala = (y1 - max_y_len / 4 < temp_Y) * (y2 + max_y_len / 4 > temp_Y)
                if np.sum(lala) != 0:
                    temp_match += 1
                    mean_Y += (y2 + y1) / 2
            if temp_match > max_match:
                max_match = temp_match
                cur_Y = mean_Y / temp_match
        if cur_Y == None: break
        group.append([])
        for i in range(len(bboxes)):
            y1, x1, y2, x2 = bboxes[i]
            lala = (y1 - max_y_len / 4 < cur_Y) * (y2 + max_y_len / 4 > cur_Y)
            if np.sum(lala) != 0 and mark[i] == 0:
                mark[i] = k
                group[k-1].append(i)
        Ys.append((k, cur_Y))
        K += 1
    Ys = sorted(Ys, key = cmp)
    print("K:",K)
    print(Ys)
    # load the weights
    # run the crops through your neural network and print them out
    import string

    letters = np.array( [str(_) for _ in range(10)] + [_ for _ in string.ascii_uppercase[:26]] + [_ for _ in string.ascii_lowercase[:11]])
    correct_num = 0
    for i in range(len(Ys)):
        k, Y = Ys[i]
        sentence = ""
        last_x2 = None
        for j in range(len(group[k-1])):
            y1, x1, y2, x2 = bboxes[group[k-1][j]]
            padx = (x2 - x1 - 1) // 6 + 1
            pady = (y2 - y1 - 1) // 6 + 1
            pady = padx = max(padx, pady)
            im2 = np.zeros((2 * pady + y2 - y1, 2 * padx + x2 - x1))
            im2[pady : y2 - y1 + pady, padx : x2 - x1 + padx] = bw[y1 : y2, x1 : x2]
            im2 = skimage.transform.resize(im2, (28, 28))
            #im2 = (im2 * 255).astype(np.uint8)
            im2 = skimage.exposure.adjust_log(im2)
            im2 = im2.astype(np.float32)
            im2 = im2.T
            #plt.imshow(im2.T, cmap = 'gray')
            im2 = im2[None,None,:,:]
            #plt.show()
            im2 = torch.from_numpy(im2)
            probs = net(im2)
            sentence += letters[probs.argmax()]
        print(sentence)
        correct_num += np.sum(np.array(list(sentence)) == np.array(list(correct_seneteces[cnt][i])))
        #print(correct_num)
    print(correct_num / len(bboxes))
    plt.show()
    cnt += 1
