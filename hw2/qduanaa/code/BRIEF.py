import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector

import matplotlib.pyplot as plt

def coordinate_encoder(x, y, patch_width):
    return x * patch_width + y
def coordinate_decoder(code, patch_width):
    y = code % patch_width
    x = code // patch_width
    x = x.astype(np.int32) - patch_width // 2
    y = y.astype(np.int32) - patch_width // 2
    return x, y
def makeTestPattern(patch_width=9, nbits=256):
    '''
    Creates Test Pattern for BRIEF

    Run this routine for the given parameters patch_width = 9 and n = 256

    INPUTS
    patch_width - the width of the image patch (usually 9)
    nbits      - the number of tests n in the BRIEF descriptor

    OUTPUTS
    compareX and compareY - LINEAR indices into the patch_width x patch_width image 
                            patch and are each (nbits,) vectors. 
    # Generate testpattern here
    x_cord = np.random.normal(mu, sigma, nbits).round().astype(np.int32)
    y_cord = np.random.normal(mu, sigma, nbits).round().astype(np.int32)
    x_cord = np.clip(x_cord, -(patch_width // 2), patch_width // 2) + patch_width // 2
    y_cord = np.clip(y_cord, -(patch_width // 2), patch_width // 2) + patch_width // 2
    compareX = coordinate_encoder(x_cord, y_cord, patch_width)
    x_cord = np.random.normal(mu, sigma, nbits).round().astype(np.int32)
    y_cord = np.random.normal(mu, sigma, nbits).round().astype(np.int32)
    x_cord = np.clip(x_cord, -(patch_width // 2), patch_width // 2) + patch_width // 2
    y_cord = np.clip(y_cord, -(patch_width // 2), patch_width // 2) + patch_width // 2
    compareY = coordinate_encoder(x_cord, y_cord, patch_width)
    compareX = compareX.reshape((nbits, 1))
    compareY = compareY.reshape((nbits, 1))
    mu, sigma = 0, patch_width / 5
    x_cord = np.random.normal(mu, sigma, (nbits, 2)).round().astype(np.int32)
    y_cord = np.random.normal(mu, sigma, (nbits, 2)).round().astype(np.int32)
    compareX = y_cord[:, 0] * patch_width + x_cord[:, 0] + (patch_width // 2) * (patch_width + 1)
    compareY = y_cord[:, 1] * patch_width + x_cord[:, 1] + (patch_width // 2) * (patch_width + 1)
    compareX = np.clip(compareX, 0, patch_width ** 2 - 1)
    compareY = np.clip(compareY, 0, patch_width ** 2 - 1)
    compareX = compareX.reshape((nbits, 1))
    compareY = compareY.reshape((nbits, 1))
    '''
    compareX = np.random.randint(patch_width*patch_width, size=(nbits, 1))
    compareY = np.random.randint(patch_width*patch_width, size=(nbits, 1))
    return  compareX, compareY

# load test pattern for Brief
test_pattern_file = '../results/testPattern.npy'
if os.path.isfile(test_pattern_file):
    # load from file if exists
    compareX, compareY = np.load(test_pattern_file)
else:
    # produce and save patterns if not exist
    compareX, compareY = makeTestPattern()
    if not os.path.isdir('../results'):
        os.mkdir('../results')
    np.save(test_pattern_file, [compareX, compareY])
    
#compareX, compareY = makeTestPattern()
def computeBrief(im, gaussian_pyramid, locsDoG, k, levels,
    compareX, compareY):
    '''
    Compute Brief feature
     INPUT
     locsDoG - locsDoG are the keypoint locations returned by the DoG
               detector.
     levels  - Gaussian scale levels that were given in Section1.
     compareX and compareY - linear indices into the 
                             (patch_width x patch_width) image patch and are
                             each (nbits,) vectors.
    
    
     OUTPUT
     locs - an m x 3 vector, where the first two columns are the image
    		 coordinates of keypoints and the third column is the pyramid
            level of the keypoints.
     desc - an m x n bits matrix of stacked BRIEF descriptors. m is the number
            of valid descriptors in the image and will vary.
    '''
    ##############################
    # compute locs, desc here
    patchsize = 9
    Hei = gaussian_pyramid.shape[0]
    Wid = gaussian_pyramid.shape[1]
    locs = []
    desc = []
    compareX = np.squeeze(compareX, axis = -1)
    compareY = np.squeeze(compareY, axis = -1)
    X_xaxis, X_yaxis = coordinate_decoder(compareX, patch_width=patchsize)
    Y_xaxis, Y_yaxis = coordinate_decoder(compareY, patch_width=patchsize)
    valid_locs = np.logical_and(locsDoG[:,0] < Wid - patchsize // 2, locsDoG[:,0] >= patchsize // 2)
    valid_locs = np.logical_and(valid_locs, np.logical_and(locsDoG[:,1] < Hei - patchsize // 2, locsDoG[:,1] >= patchsize // 2))
    locs = locsDoG[valid_locs]
    X_coord = np.expand_dims(locs[:,0], axis = -1)
    Y_coord = np.expand_dims(locs[:,1], axis = -1)
    X_xaxis = X_coord + X_xaxis
    X_yaxis = Y_coord + X_yaxis
    Y_xaxis = X_coord + Y_xaxis
    Y_yaxis = Y_coord + Y_yaxis
    level = np.expand_dims(locs[:,2] + 1, axis = -1)
    X_xaxis = X_xaxis.astype(np.int16)
    X_yaxis = X_yaxis.astype(np.int16)
    Y_xaxis = Y_xaxis.astype(np.int16)
    Y_yaxis = Y_yaxis.astype(np.int16)
    level = level.astype(np.int16)
    X_mask = gaussian_pyramid[X_yaxis, X_xaxis, level]
    Y_mask = gaussian_pyramid[Y_yaxis, Y_xaxis, level]
    desc = X_mask < Y_mask
    desc = desc.astype(np.int16)
    return locs, desc



def briefLite(im):
    '''
    INPUTS
    im - gray image with values between 0 and 1

    OUTPUTS
    locs - an m x 3 vector, where the first two columns are the image coordinates 
            of keypoints and the third column is the pyramid level of the keypoints
    desc - an m x n bits matrix of stacked BRIEF descriptors. 
            m is the number of valid descriptors in the image and will vary
            n is the number of bits for the BRIEF descriptor
    '''
    ###################
    # TO DO ...
    locs, gauss_pyr = DoGdetector(im)
    locs, desc = computeBrief(im, gauss_pyr, locs, 0, [-1, 0, 1, 2, 3, 4], compareX=compareX, compareY=compareY)
    return locs, desc

def briefMatch(desc1, desc2, ratio=0.8):
    '''
    performs the descriptor matching
    inputs  : desc1 , desc2 - m1 x n and m2 x n matrix. m1 and m2 are the number of keypoints in image 1 and 2.
                                n is the number of bits in the brief
    outputs : matches - p x 2 matrix. where the first column are indices
                                        into desc1 and the second column are indices into desc2
    '''
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')
    # find smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)
    # find second smallest distance
    d12 = np.partition(D, 2, axis=1)[:,0:2]
    d2 = d12.max(1)
    r = d1/(d2+1e-10)
    is_discr = r<ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]

    matches = np.stack((ix1,ix2), axis=-1)
    return matches

def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()
    # draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')
    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i,0], 0:2]
        pt2 = locs2[matches[i,1], 0:2].copy()
        pt2[0] += im1.shape[1]
        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])
        plt.plot(x,y,'r', linewidth = 0.25)
        plt.plot(x,y,'g.', linewidth = 0.25)
    plt.show()
    
    

if __name__ == '__main__':
    # test makeTestPattern
    compareX, compareY = makeTestPattern()
    # test briefLite
    #im = cv2.imread('../data/chickenbroth_01.jpg')
    #im = cv2.imread('../data/incline_L.png')
    im = cv2.imread('../data/pf_scan_scaled.jpg')

    locs, desc = briefLite(im)  
    #fig = plt.figure()
    #plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cmap='gray')
    #plt.plot(locs[:,0], locs[:,1], 'r.')
    #plt.draw()
    #plt.waitforbuttonpress(0)
    #plt.close(fig)
    # test matches
    #im1 = cv2.imread('../data/model_chickenbroth.jpg')
    #im2 = cv2.imread('../data/chickenbroth_01.jpg')
    #im1 = cv2.imread('../data/incline_L.png')
    #im2 = cv2.imread('../data/incline_R.png')
    im1 = cv2.imread('../data/pf_scan_scaled.jpg')
    im2 = cv2.imread('../data/pf_stand.jpg')
    #im2 = im1
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    plotMatches(im1,im2,matches,locs1,locs2)

