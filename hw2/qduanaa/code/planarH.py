import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    N = p1.shape[1]
    A = np.zeros((2 * N, 9))
    A[0::2,:2] = np.transpose(-p2)
    A[0::2,2] = -1
    A[1::2, 3:5] = -p2.T
    A[1::2, 5] = -1
    A[::2, -3:] = np.expand_dims(p1.T[:,0], axis = -1)
    A[1::2, -3:] = np.expand_dims(p1.T[:,1], axis = -1)
    A[::2, -3:-1] = A[::2, -3:-1] * p2.T
    A[1::2, -3:-1] = A[1::2, -3:-1] * p2.T
    AA = np.matmul(A.T, A).astype(np.float64)
    w, v = np.linalg.eig(AA)
    index = (w == np.amin(w))
    h = np.squeeze(v[:,index])
    H2to1 = h.reshape((3, 3))
    H2to1 = H2to1 / H2to1[-1, -1]
    return H2to1
#def Dot(x, y):
    #x = np.expand_dims(x, axis = -2)
    #y = np.expand_dims(y, axis = -1)
    #return np.matmul(x, y)
def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    i = 0
    best_numInlier = 0
    bestH = None
    index = np.arange(matches.shape[0])
    while i < num_iter:
        tempInliers = np.random.choice(index, size = 4, replace = False)
        tempH = computeH(locs1[matches[tempInliers,0], :-1].T, locs2[matches[tempInliers,1], :-1].T)
        vec_x = locs1[matches[:,0]].astype(np.float64)
        vec_u = locs2[matches[:,1]].astype(np.float64)
        vec_x[:,-1] = 1
        vec_u[:,-1] = 1
        vec_u = np.expand_dims(vec_u, axis = -1)
        vec_u = np.matmul(tempH, vec_u)
        vec_u = np.squeeze(vec_u, axis = -1)
        #lamb = Dot(vec_u, vec_x) / Dot(vec_x, vec_x)
        #lamb = np.squeeze(lamb, axis = -1)
        diff = vec_x - vec_u / np.expand_dims(vec_u[:,-1], axis =- 1)
        dis = np.linalg.norm(diff, axis = 1)
        temp_len =  np.sum(dis < tol)
        save = tempInliers
        temp_diff = diff[save]
        temp_dis = dis[save]
        tempInliers = index[dis < tol]
        #assert(temp_len >= 4)
        if temp_len == len(index):continue
        if temp_len > best_numInlier:
            best_numInlier = temp_len 
            bestH = tempH
            print(i, best_numInlier / len(index))
        i = i + 1
    return bestH
def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    outshape = im2.shape[:-1]
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = cv2.warpPerspective(im2, H2to1, (outshape[1] * 2, outshape[0] * 2))
    print(H2to1)
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return pano_im

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    im3 = generatePanorama(im1, im2)
