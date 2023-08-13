import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
def LucasKanadeBasis(It, It1, rect, bases):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	bases: [n, m, k] where nxm is the size of the template.
	# Output:
	#	p: movement vector [dp_x, dp_y]

    # Put your implementation here

	p = np.zeros(2)
	splinet = RectBivariateSpline(np.arange(It.shape[1]), np.arange(It.shape[0]), It.T)
	splinet1 = RectBivariateSpline(np.arange(It1.shape[1]), np.arange(It1.shape[0]), It1.T)
	x = np.linspace(rect[0], rect[2], bases.shape[1])
	y = np.linspace(rect[1], rect[3], bases.shape[0])
	xgrid, ygrid = np.meshgrid(x, y)
	T = splinet.ev(xgrid, ygrid)
	bases = bases.reshape(-1, bases.shape[-1])
	proj = np.eye(bases.shape[0]) - bases @ bases.T
	while True:
		I = splinet1.ev(xi = xgrid + p[0], yi = ygrid + p[1])
		Ix = cv2.Sobel(I, ddepth = -1, dx = 1, dy = 0, ksize = 3).reshape(-1)
		Iy = cv2.Sobel(I, ddepth = -1, dx = 0, dy = 1, ksize = 3).reshape(-1)
		A = np.zeros((Ix.shape[0], 2))
		A[:,0] = Ix
		A[:,1] = Iy
		b = (T - I).reshape(-1)
		A = proj @ A
		b = proj @ b
		delta_p = np.linalg.inv(A.T @ A) @ np.matmul(A.T, b)
		p = p + delta_p
		mag = np.sum(delta_p ** 2)
		if mag < 1e-5 :
			break
	return p
