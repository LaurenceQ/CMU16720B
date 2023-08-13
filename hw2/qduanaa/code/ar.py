import numpy as np
import cv2
import matplotlib.pyplot as plt
from planarH import computeH
def compute_extrinsics(K, H):
    phi = np.matmul(np.linalg.inv(K), H)
    u, s, vh = np.linalg.svd(phi[:, :-1], full_matrices = True)
    w = np.matmul(u[:, :-1], vh)
    w3 = np.cross(w[:,0], w[:,1])
    w = np.stack((w[:,0], w[:,1], w3), axis = -1)
    # need Experiment to determine the sign of w3!!!
    if np.linalg.det(w) > 0 : 
        w[:, -1] = w[:, -1] * -1
    #w[:, -1] = w[:, -1] * -1
    lamb = np.sum(phi[:,:-1] / w[:,:-1]) / 6
    t = phi[:, -1] / lamb
    return w, t

    # eliminate reflection component in the Rotation matrix
def project_extrinsics(K, W, R, t):
    X = np.matmul(R, W) + np.expand_dims(t, axis = -1)
    X = np.matmul(K, X)
    print(X)
    X /= X[-1, :]
    return X
if __name__ == '__main__':
    K = [[3043.72, 0, 1196], [0, 3043.72, 1604], [0, 0, 1]]
    K = np.array(K, dtype = np.float32)
    W = [[0, 18.2, 18.2, 0], [0, 0, 26, 26], [0, 0, 0, 0]]
    W = np.array(W, dtype = np.float32)
    D = [[483, 1704, 2175, 67], [810, 781, 2217, 2286]]
    D = np.array(D, dtype = np.float32)
    H = computeH(D, W[:-1,:])
    R, t = compute_extrinsics(K, H)
    points = np.loadtxt('../data/sphere.txt', dtype=np.float32)
    points[2] = points[2] + 6.8581 / 2
    points[0] = points[0] + 6.2
    points[1] = points[1] + 18.4
    X = project_extrinsics(K, points, R, t)
    im = cv2.imread('../data/prince_book.jpeg')
    imgplot = plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.plot(X[0,:], X[1,:], color = 'y', linewidth=0.7)
    plt.show()