'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import submission as sb
import numpy as np
import matplotlib.pyplot as plt
import helper
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')
data = np.load('../data/some_corresp.npz')
pts1 = data['pts1']
pts2 = data['pts2']
M = 640
F, inlier = sb.ransacF(pts1, pts2, M)
pts1 = pts1[inlier,:]
pts2 = pts2[inlier,:]

F8 = sb.eightpoint(pts1,pts2, M)

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
    P, error = sb.triangulate(C1, pts1, C2, pts2)
    if np.sum(P[:,-1] < 0) > 0:
        continue
    else : break
P, err = sb.triangulate(C1, pts1, C2, pts2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(P[:,0], P[:,1], P[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
M2, P = sb.bundleAdjustment(K1, M1, pts1, K2, M2, pts1, P)
fig = plt.figure()
ax2 = fig.add_subplot(projection='3d')
ax2.scatter(P[:,0], P[:,1], P[:,2])
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.show()
