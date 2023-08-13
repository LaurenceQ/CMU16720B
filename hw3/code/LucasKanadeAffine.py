import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
import cv2
def LucasKanadeAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here
	M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
	Jacob = np.zeros((It.shape[0], It.shape[1], 2, 6))
	x = np.arange(It1.shape[1])
	y = np.arange(It1.shape[0])
	Jacob[:,:,0,0] = x
	Jacob[:,:,0,1] = y[:, None]
	Jacob[:,:,0,2] = 1
	Jacob[:,:,1,3] = x 
	Jacob[:,:,1,4] = y[:, None]
	Jacob[:,:,1,5] = 1
	Jacob = Jacob.reshape((-1, 2, 6))
	#Ix = cv2.Sobel(It1, ddepth = -1, dx = 1, dy = 0, ksize = 3)
	#Iy = cv2.Sobel(It1, ddepth = -1, dx = 0, dy = 1, ksize = 3)
	#It_spline = RectBivariateSpline(x, y, It1.T)
	#xgrid, ygrid = np.meshgrid(x, y)
	#Ix = It_spline.ev(xgrid, ygrid, dx=1, dy=0)
	#Iy = It_spline.ev(xgrid, ygrid, dx=0, dy=1)
	Ix = cv2.Sobel(It1, ddepth = -1, dx = 1, dy = 0, ksize = 3)
	Iy = cv2.Sobel(It1, ddepth = -1, dx = 0, dy = 1, ksize = 3)
	'''fig, axes = plt.subplots(1, 4)
	axes[0].imshow(Ix, cmap = 'gray')
	axes[1].imshow(Iy, cmap = 'gray')
	axes[2].imshow(Ix_grad, cmap = 'gray')
	axes[3].imshow(Iy_grad, cmap = 'gray')
	plt.show()
	exit(0)'''
	while True:
		I = affine_transform(It1, M)
		Ix_warp = affine_transform(Ix, M).reshape(-1)
		Iy_warp = affine_transform(Iy, M).reshape(-1)
		valid = affine_transform(np.ones(It1.shape), M)
		A = np.zeros((Ix.shape[0] * Ix.shape[1], 2))
		A[:,0] = Ix_warp.reshape(-1)
		A[:,1] = Iy_warp.reshape(-1)
		A = np.matmul(A[:,None,:], Jacob)
		A = A.squeeze()
		b = (It * valid - I).reshape(-1)
		delta_p = np.linalg.inv(A.T @ A) @ np.matmul(A.T, b)
		M = M + delta_p.reshape(M.shape) 
		mag = np.sqrt(np.sum(delta_p ** 2))
		if mag < 0.15:
			break
	return M 