import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    outshape = (im2.shape[0], im1.shape[1] + im2.shape[1])
    H1 = im1.shape[0]
    W1 = im1.shape[1]
    pano_im = cv2.warpPerspective(im2, H2to1, (outshape[1], outshape[0]))
    pano_im[:,:W1] = np.maximum(pano_im[:,:W1], im1)
    return pano_im


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    H1, W1, _ = im1.shape
    H2, W2, _ = im2.shape
    out_size = np.zeros((2,))
    out_size[0] = max(W1, W2) * 2
    v1 = np.matmul(H2to1, np.array([0, 0, 1]))
    v2 = np.matmul(H2to1, np.array([W2, 0, 1]))
    v3 = np.matmul(H2to1, np.array([0, H2, 1]))
    v4 = np.matmul(H2to1, np.array([W2, H2, 1]))
    v1 /= v1[-1]
    v2 /= v2[-1]
    v3 /= v3[-1]
    v4 /= v4[-1]
    minX = min((0, v1[0], v2[0], v3[0], v4[0]))
    maxX = max((W1, v1[0], v2[0], v3[0], v4[0]))
    minY = min((0, v1[1], v2[1], v3[1], v4[1]))
    maxY = max((H1, v1[1], v2[1], v3[1], v4[1]))
    out_size[1] = np.round((maxY - minY) / (maxX - minX) * out_size[0])
    out_size = out_size.astype(np.int32)
    M = np.zeros((3, 3))
    M[0][0] = M[1][1] = M[2][2] = 1
    M[0][2] = max(0, -minX)
    M[1][2] = max(0, -minY)
    warp_im1 = cv2.warpPerspective(im1, M, out_size)
    warp_im2 = cv2.warpPerspective(im2, np.matmul(M, H2to1), out_size)
    pano_im = np.maximum(warp_im1, warp_im2)
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000 , tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    return pano_im
if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    pano_im = generatePanorama(im1, im2)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()