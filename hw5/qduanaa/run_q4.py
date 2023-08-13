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

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
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
cnt = 0
for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    #plt.imshow(im1)
    #plt.show()
    #findLetters(im1)
    bboxes, bw = findLetters(im1)
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
        #print("k:",k)
        #print(cur_Y)
        #print(max_match)
    Ys = sorted(Ys, key = cmp)
    #print("K:",K)
    #print(Ys)
    #print(np.sum(mark != 0))
    # find the rows using..RANSAC, counting, clustering, etc.
    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
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
            #padx = pady = 10
            im2 = np.ones((2 * pady + y2 - y1, 2 * padx + x2 - x1))
            im2[pady : y2 - y1 + pady, padx : x2 - x1 + padx] = bw[y1 : y2, x1 : x2]
            im2 = skimage.transform.resize(im2, (32, 32))
            im2 = (im2 * 255).astype(np.uint8)
            im2 = skimage.exposure.equalize_hist(im2, nbins=256, mask=None)
            im2 = im2.astype(np.float32)
            im2 *= 0.95
            #plt.imshow(im2, 'gray')
            #plt.show()
            im2 = im2.T
            #im2 = (im2 == 1).astype(np.float32)
            im2 = im2.reshape(1, -1)
            h1 = forward(im2,params,'layer1')
            probs = forward(h1,params,'output',softmax)
            #if last_x2 != None and x1 > last_x2 + 1.5 * max_x_len:
                #sentence += ' '
            last_x2 = x2
            sentence += letters[probs.argmax()]
            #im2 = im2.reshape(32, 32).T
            #print(sentence[-1])
            #print("box len:", y2 - y1, x2 - x1)
            #print("im2 shape:", im2.shape)
            #plt.pause(5)
        print(sentence)
        correct_num += np.sum(np.array(list(sentence)) == np.array(list(correct_seneteces[cnt][i])))
        #print(correct_num)
    print(correct_num / len(bboxes))
    plt.show()
    cnt += 1