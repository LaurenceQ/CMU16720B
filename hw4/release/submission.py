"""
Homework4.
Replace 'pass' by your implementation.
"""
import numpy as np
import cv2 
import helper
from scipy.optimize import leastsq
# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation
    T = np.zeros((3, 3))
    T[0, 0] = 2 / M
    T[1, 1] = 2 / M
    T[0, 2] = -1
    T[1, 2] = -1
    T[2, 2] = 1
    N = pts1.shape[0]
    normed_x1 = np.ones((N, 3))
    normed_x1[:, :-1] = pts1
    normed_x1 = np.matmul(T, normed_x1.T).T
    normed_x2 = np.ones((N, 3))
    normed_x2[:, :-1] = pts2
    normed_x2 = np.matmul(T, normed_x2.T).T
    A = np.array([
        normed_x1[:,0] * normed_x2[:,0], normed_x1[:,0] * normed_x2[:,1], normed_x1[:,0],
        normed_x1[:,1] * normed_x2[:,0], normed_x1[:,1] * normed_x2[:,1], normed_x1[:,1],
        normed_x2[:,0], normed_x2[:,1], np.ones(N) 
        ])
    A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    min_ind = np.where(np.min(S) == S)
    F = np.reshape(Vh[min_ind[-1], :], (3, 3))
    U, S, Vh = np.linalg.svd(F, full_matrices=False)
    S[-1] = 0
    F = np.dot(U * S, Vh)
    F = helper.refineF(F, normed_x1[:,:-1], normed_x2[:,:-1])
    F = T.T @ F @ T

    return F

'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def calc_coef(x1, x5, x6, x8, x9, y1, y5, y6, y8, y9):
    coef = np.zeros(4)
    t0 = (x5 - y5) * (x9 - y9) - (x6 - y6) * (x8 - y8)
    t1 = (x9 - y9) * y5 + (x5 - y5) * y9 - (x6 - y6) * y8 - (x8 - y8) * y6
    t2 = y9 * y5 - y6 * y8
    coef[0] = (x1 - y1) * t0
    coef[1] = (x1 - y1) * t1 + y1 * t0
    coef[2] = (x1 - y1) * t2 + y1 * t1
    coef[3] = y1 * t2
    return coef
def sevenpoint(pts1, pts2, M):
    T = np.zeros((3, 3))
    T[0, 0] = 2 / M
    T[1, 1] = 2 / M
    T[0, 2] = -1
    T[1, 2] = -1
    T[2, 2] = 1
    N = pts1.shape[0]
    normed_x1 = np.ones((N, 3))
    normed_x1[:, :-1] = pts1
    normed_x1 = np.matmul(T, normed_x1.T).T
    normed_x2 = np.ones((N, 3))
    normed_x2[:, :-1] = pts2
    normed_x2 = np.matmul(T, normed_x2.T).T
    A = np.array([
        normed_x1[:,0] * normed_x2[:,0], normed_x1[:,0] * normed_x2[:,1], normed_x1[:,0],
        normed_x1[:,1] * normed_x2[:,0], normed_x1[:,1] * normed_x2[:,1], normed_x1[:,1],
        normed_x2[:,0], normed_x2[:,1], np.ones(N) 
        ])
    A = A.T
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    F1 = Vh[-2,:].reshape(3, 3)
    F2 = Vh[-1,:].reshape(3, 3)
    coef = np.zeros(4)
    coef = coef + calc_coef(F1[0][0], F1[1][1], F1[1][2], F1[2][1], F1[2][2], F2[0][0], F2[1][1], F2[1][2], F2[2][1], F2[2][2])
    coef = coef + calc_coef(-F1[0][1], -F1[1][0], -F1[1][2], -F1[2][0], -F1[2][2], -F2[0][1], -F2[1][0], -F2[1][2], -F2[2][0], -F2[2][2])
    coef = coef + calc_coef(F1[0][2], F1[1][0], F1[1][1], F1[2][0], F1[2][1], F2[0][2], F2[1][0], F2[1][1], F2[2][0], F2[2][1])
    roots = np.roots(coef)
    F = []
    for alpha in roots :
        if np.iscomplex(alpha): continue
        tmp = alpha * F1 + (1 - alpha) * F2
        tmp = T.T @ tmp @ T
        F.append(tmp)
    F = np.array(F)
    return F


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    return np.matmul(K2.T, np.matmul(F, K1))


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    A = np.zeros((N, 4, 4))
    A[:,0,:] = C1[2].T * pts1[:,1,None] - C1[1].T
    A[:,1,:] = C1[0].T - pts1[:,0,None] * C1[2].T
    A[:,2,:] = C2[2].T * pts2[:,1,None] - C2[1].T
    A[:,3,:] = C2[0].T - pts2[:,0,None] * C2[2].T
    U, S, Vh = np.linalg.svd(A)
    X = Vh[:,-1,:]
    X = X / X[:,-1,None]
    x1 = np.matmul(C1, X.T).T
    x1 = x1 / x1[:,-1,None]
    x2 = np.matmul(C2, X.T).T
    x2 = x2 / x2[:,-1,None]
    err = np.sum(np.square(pts1 - x1[:,:-1]) + np.square(pts2 - x2[:,:-1]))
    return X[:,:-1], err
'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    point = np.array([x1, y1, 1])
    line = np.matmul(F, point)
    wid_sz = 11
    sigma = 4
    search_range = 41
    gauss = ((np.arange(wid_sz) - wid_sz // 2) ** 2)[:,None] + ((np.arange(wid_sz) - wid_sz // 2) ** 2)[None,:]
    gauss = np.exp(-.5 * gauss / (sigma ** 2))
    gauss = gauss[:,:,None]
    gauss = gauss / np.sum(gauss)
    im1_patch = im1[y1-wid_sz//2:y1+wid_sz//2+1, x1-wid_sz//2:x1+wid_sz//2+1]
    min_loss = 1e30
    minx = -1
    miny = -1
    a = im2.shape[0]
    k = -line[0] / line[1]
    x2_min = 0
    x2_max = im2.shape[1]
    if k < 0 :
        x2_min = a/k + line[2] / k / line[1]
        x2_max = line[2] / k / line[1]
    else:
        x2_min = line[2] / k / line[1]
        x2_max = a / k + line[2] / line[1] / k
    epsilon = x2_max - x2_min
    epsilon = epsilon / 200
    x2_min = round(x2_min / epsilon)
    x2_max = round(x2_max / epsilon)
    for i in range(x2_min, x2_max):
        x2 = i * epsilon
        y2 = -line[2] / line[1] - line[0] / line[1] * x2
        if y2 < y1 - search_range or y2 > y1 + search_range:continue
        y2 = np.round(y2).astype(np.int32)
        x2 = np.round(x2).astype(np.int32)
        im2_patch = im2[y2-wid_sz//2:y2+wid_sz//2 + 1, x2-wid_sz//2:x2+wid_sz//2 + 1]
        if im2_patch.shape[0] == im1_patch.shape[0] and im2_patch.shape[1] == im1_patch.shape[1]:
            loss = np.sum(((im1_patch - im2_patch)**2) * gauss)
            if loss < min_loss:
                min_loss = loss
                minx, miny = x2, y2
    return minx, miny

'''
Q5.1: RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M):
    minError = 1e9
    iter_num = 2000
    threshold = 0.015
    M = 640
    N = pts1.shape[0]
    index = np.arange(N)
    maxInlier = np.zeros(N)
    homo_p1 = np.ones((N, 3))
    homo_p1[:,:-1] = pts1
    homo_p2 = np.ones((N, 3))
    homo_p2[:,:-1] = pts2
    #save_indices = None
    for _ in range(iter_num):
        indices = np.random.choice(index, size = 7, replace = False)
        P1 = pts1[indices,:]
        P2 = pts2[indices,:]
        Fs = sevenpoint(P1, P2, M)
        for F in Fs:
            dis = np.abs(np.sum(homo_p1[:,:] * (F @ homo_p2.T).T, axis = 1))
            inlier = dis < threshold
            if np.sum(inlier) > np.sum(maxInlier):
                maxInlier = inlier
                minError = np.sum(dis)
                #save_indices = indices
            elif np.sum(inlier) == np.sum(maxInlier) and np.sum(dis) < minError:
                maxInlier = inlier
                minError = np.sum(dis)
                #save_indices = indices
    pts1 = pts1[maxInlier,:]
    pts2 = pts2[maxInlier,:]
    refineF = eightpoint(pts1, pts2, M)
    dis = np.abs(np.sum(homo_p1[:,:] * (refineF @ homo_p2.T).T, axis = 1))
    maxInlier = dis < threshold

    return refineF, maxInlier

'''
Q5.2: Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        k = r / theta
        K = np.array(
            [[0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]]
        )
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        return R

'''
Q5.2: Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    tolerance = 1e-15
    A = (R - R.T)/2
    rho = np.array(A[[2, 0, 1], [1, 2, 0]])[:, None]
    norm = np.float32(np.linalg.norm(rho))
    c = np.float32((np.trace(R) - 1)/2)
    if norm < tolerance and (c - 1) < tolerance:
        r = np.array([0.0, 0.0, 0.0])[:, None]
        return r
    elif norm < 1e-15 and (c + 1) < tolerance:
        v = None
        for i in range(R.shape[-1]):
            v = (R + np.eye(3))[:, i]
            if np.count_nonzero(v) > 0:
                break
        u = v/np.linalg.norm(v)
        r = (u*np.pi)[:, None]
        if np.linalg.norm(r) == np.pi and (r[0, 0] == r[1, 0] == 0 and r[2, 0] < 0.0) or (r[0, 0] == 0 and r[1, 0] < 0) or (r[0, 0] < 0):
            return -r
        else:
            return r
    else:
        u = rho/norm
        theta = np.arctan2(norm, c)
        r = u*theta
        return r
'''
Q5.3: Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    r2 = x[-6:-3]
    t2 = x[-3:]
    P = x[:-6].reshape(-1, 3)
    R2 = rodrigues(r2)
    M2 = np.zeros((3, 4))
    M2[:,:-1] = R2
    M2[:,-1] = t2
    C1 = K1 @ M1
    C2 = K2 @ M2
    N = P.shape[0]
    X = np.ones((N, 4))
    X[:,:-1] = P
    x1 = np.matmul(C1, X.T).T
    x1 = x1 / x1[:,-1,None]
    x2 = np.matmul(C2, X.T).T
    x2 = x2 / x2[:,-1,None]
    return np.concatenate(((p1 - x1[:,:-1]).reshape([-1]), (p2 - x2[:,:-1]).reshape([-1])))[:,None]

'''
Q5.3 Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def cost_func(x, K1, M1, p1, K2, p2):
    residual = rodriguesResidual(K1, M1, p1, K2, p2, x)

    return np.squeeze(residual)
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    x = np.concatenate((P_init.reshape([-1]), invRodrigues(M2_init[:,:-1]).reshape([-1]), M2_init[:,-1].reshape([-1])))
    bestx, _ = leastsq(cost_func, x, args = (K1, M1, p1, K2, p2))
    #print(_)
    r2 = bestx[-6:-3]
    t2 = bestx[-3:]
    P = bestx[:-6].reshape(-1, 3)
    R2 = rodrigues(r2)
    M2 = np.zeros((3, 4))
    M2[:,:-1] = R2
    M2[:,-1] = t2
    return M2, P
