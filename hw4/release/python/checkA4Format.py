"""
Check the dimensions of function arguments
This is *not* a correctness check

Written by Chen Kong, 2018.
"""
import numpy as np
import matplotlib.pyplot as plt
import submission as sub
import cv2
import helper
data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

M = 640
# 2.1
N = data['pts1'].shape[0]
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
assert F8.shape == (3, 3), 'eightpoint returns 3x3 matrix'
# 2.2

ind = [105, 107,  88,  30,  33,  79,  26]
pts1 = data['pts1'][ind,:]
pts2 = data['pts2'][ind,:]
F7 = sub.sevenpoint(data['pts1'][ind, :], data['pts2'][ind, :], M)
#np.savez('q2_2.npz', F = F7[-1], M = M, pts1 = pts1, pts2 = pts2)
assert (len(F7) == 1) | (len(F7) == 3), 'sevenpoint returns length-1/3 list'
for f7 in F7:
    #helper.displayEpipolarF(im1, im2, f7) 
    #print(f7)
    assert f7.shape == (3, 3), 'seven returns list of 3x3 matrix'
# 3.1
C1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
C2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
P, err = sub.triangulate(C1, data['pts1'], C2, data['pts2'])
print(err)
assert P.shape == (N, 3), 'triangulate returns Nx3 matrix P'
assert np.isscalar(err), 'triangulate returns scalar err'

# 4.1
x2, y2 = sub.epipolarCorrespondence(im1, im2, F8, data['pts1'][0, 0], data['pts1'][0, 1])
assert np.isscalar(x2) & np.isscalar(y2), 'epipolarCoorespondence returns x & y coordinates'
#pts1, pts2 = helper.epipolarMatchGUI(im1, im2, F8)
#print(pts1, pts2)
#np.savez("q4_1.npz", F= F8, pts1 = pts1, pts2 = pts2)
# 5.1

data = np.load('../data/some_corresp_noisy.npz')
F, inliers = sub.ransacF(data['pts1'], data['pts2'], M)
assert F.shape == (3, 3), 'ransacF returns 3x3 matrix'
pts1 = data['pts1']
pts2 = data['pts2']

# Compute fundamental matrix using eight-point method
F_eight = sub.eightpoint(pts1, pts2, M)

# Compute fundamental matrix using RANSAC method
F_ransac, inliers = sub.ransacF(pts1, pts2, M)
helper.displayEpipolarF(im1, im2, F_ransac)
helper.displayEpipolarF(im1, im2, F_eight)
# 5.2
r = np.ones([3, 1])
R = sub.rodrigues(r)
assert R.shape == (3, 3), 'rodrigues returns 3x3 matrix'

R = np.eye(3);
r = sub.invRodrigues(R)
assert (r.shape == (3, )) | (r.shape == (3, 1)), 'invRodrigues returns 3x1 vector'

# 5.3
K1 = np.random.rand(3, 3)
K2 = np.random.rand(3, 3)
M1 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
M2 = np.concatenate([np.random.rand(3, 3), np.ones([3, 1])], axis=1)
r2 = np.ones(3)
t2 = np.ones(3)
x = np.concatenate([P.reshape([-1]), r2, t2])
residuals = sub.rodriguesResidual(K1, M1, data['pts1'], K2, data['pts1'], x)
assert residuals.shape == (4 * N, 1), 'rodriguesResidual returns vector of size 4Nx1'

M2, P = sub.bundleAdjustment(K1, M1, data['pts1'], K2, M2, data['pts1'], P)
assert M2.shape == (3, 4), 'bundleAdjustment returns 3x4 matrix M'
assert P.shape == (N, 3), 'bundleAdjustment returns Nx3 matrix P'

exit(0)
print('Format check passed.')
