import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy.ndimage import affine_transform

def InverseCompositionAffine(It, It1):
	# Input: 
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here
	M33 = np.eye(3)
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
	Tx = cv2.Sobel(It, ddepth = -1, dx = 1, dy = 0, ksize = 3)
	Ty = cv2.Sobel(It, ddepth = -1, dx = 0, dy = 1, ksize = 3)
	A = np.zeros((Tx.shape[0] * Tx.shape[1], 2))
	A[:,0] = Tx.reshape(-1)
	A[:,1] = Ty.reshape(-1)
	A = np.matmul(A[:,None,:], Jacob)
	A = A.squeeze()
		
	while True:
		I = affine_transform(It1, M33).reshape(-1)
		valid = affine_transform(np.ones(It1.shape), M33).reshape(-1)
		#print(valid.sum())
		b = (I - It.reshape(-1) * valid).reshape(-1)
		delta_p = np.linalg.inv(A.T @ A) @ np.matmul(A.T, b)
		if np.sum(delta_p ** 2) == 0:
			print(A[:20])
			print(b[:20])
			print(b.max())
			print(delta_p)
			break
		delta_M = np.eye(3)
		delta_M[:2,:] += delta_p.reshape((2, 3))
		#print("deltaM:")
		#print(delta_M)
		delta_M = np.linalg.inv(delta_M)
		mag = np.linalg.norm(delta_p)
		#print(mag)
		if mag < 0.15:
			break	
		#print("delta_p")
		#print(delta_p)
		M33 = M33 @ delta_M
	#print(M33)
	return M33[:2,:]
