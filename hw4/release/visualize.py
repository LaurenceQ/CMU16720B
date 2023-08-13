'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import submission as sb
import numpy as np
import matplotlib.pyplot as plt
import helper
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = 640
F8 = sb.eightpoint(data['pts1'], data['pts2'], M)
intrK = np.load('../data/intrinsics.npz')
K1 = intrK['K1']
K2 = intrK['K2']
E = sb.essentialMatrix(F8, K1, K2)
M2s = helper.camera2(E)
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
    P, error = sb.triangulate(C1, data['pts1'], C2, data['pts2'])
    if np.sum(P[:,-1] < 0) > 0:
        continue
    else : break
dataIm1 = np.load("./../data/templeCoords.npz")
x1 = dataIm1['x1']
y1 = dataIm1['y1']
N = x1.shape[0]
x2 = np.zeros(N, np.int32)
y2 = np.zeros(N, np.int32)
x1 = np.squeeze(x1)
y1 = np.squeeze(y1)
for i in range(N):
    x2[i], y2[i] = sb.epipolarCorrespondence(im1, im2, F8, x1[i], y1[i])
pts1 = np.stack((x1, y1), axis = -1)
pts2 = np.stack((x2, y2), axis = -1)
P, err = sb.triangulate(C1, pts1, C2, pts2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2])
#np.savez("q4_2.npz", F = F8, M1=M1, M2 = M2, C1 = C1, C2 = C2)
plt.show()