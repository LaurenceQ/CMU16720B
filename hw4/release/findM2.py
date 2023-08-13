'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import cv2 
from helper import camera2 
from submission import essentialMatrix, eightpoint, triangulate
M = 640
data = np.load('../data/some_corresp.npz')
N = data['pts1'].shape[0]
F = eightpoint(data['pts1'], data['pts2'], M)
intrK = np.load('../data/intrinsics.npz')
K1 = intrK['K1']
K2 = intrK['K2']
E = essentialMatrix(F, K1, K2)
M2s = camera2(E)
M1 = np.zeros((3, 4))
M1[0][0] = M1[1][1] = M1[2][2] = 1
C1 = K1 @ M1
C2 = None
M2 = None
for i in range(4):
    M2 = M2s[:,:,i]
    R = M2[:, :-1]
    if np.linalg.det(R) == -1:
        continue
    C2 = K2 @ M2
    P, error = triangulate(C1, data['pts1'], C2, data['pts2'])
    if np.sum(P[:,-1] < 0) > 0:
        continue
    else : break
#np.savez('q3_3.npz', M2 = M2, C2 = C2, P = P)