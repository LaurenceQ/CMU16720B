import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
#from skimage.measure import compare_psnr as psnr
from skimage.metrics import peak_signal_noise_ratio as psnr

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

# do PCA
dim = 32
u, s, vh = np.linalg.svd(train_x, full_matrices = False)
#s = np.sort(s)
dim_s = np.zeros((dim, s.shape[0]))
dim_s[:dim, :dim] = np.eye(dim)
proj = dim_s @ vh
# rebuild a low-rank version
lrank = (proj @ train_x.T).T
# rebuild it
recon = (vh.T[:,:dim]) @ (lrank.T)
recon = recon.T
#s[dim:] = 0
#recon = (u * s) @ vh
'''
for i in range(5):
    plt.subplot(2,1,1)
    plt.imshow(train_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon[i].reshape(32,32).T)
    plt.show()
'''
# build valid dataset

recon_valid = (vh.T[:,:dim] @ proj @ valid_x.T).T
total = []
valid_y = valid_data['valid_labels'].argmax(1)
cnt = np.zeros(5)
for i in range(len(valid_x)):
    if valid_y[i] >= 5:continue
    if cnt[valid_y[i]] == 2: continue
    cnt[valid_y[i]] += 1
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon_valid[i].reshape(32,32).T)
    plt.show()
for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())