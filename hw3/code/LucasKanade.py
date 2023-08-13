import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input: 
	#	It: template image
	#	It1: Current image
	#	rect: Current position of the car
	#	(top left, bot right coordinates)
	#	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	#	p: movement vector [dp_x, dp_y]
	
    # Put your implementation here
	p = p0
	splinet = RectBivariateSpline(np.arange(It.shape[1]), np.arange(It.shape[0]), It.T)
	splinet1 = RectBivariateSpline(np.arange(It1.shape[1]), np.arange(It1.shape[0]), It1.T)
	x = np.linspace(rect[0], rect[2], round(rect[2] - rect[0] + 1))
	y = np.linspace(rect[1], rect[3], round(rect[3] - rect[1] + 1))
	#ref = It1[rect[1]:rect[3]+1, rect[0]:rect[2]+1]
	xgrid, ygrid = np.meshgrid(x, y)
	T = splinet.ev(xgrid, ygrid)
	while True:
		I = splinet1.ev(xi = xgrid + p[0], yi = ygrid + p[1])
		Ix = cv2.Sobel(I, ddepth = -1, dx = 1, dy = 0, ksize = 3).reshape(-1)
		Iy = cv2.Sobel(I, ddepth = -1, dx = 0, dy = 1, ksize = 3).reshape(-1)
		A = np.zeros((Ix.shape[0], 2))
		A[:,0] = Ix
		A[:,1] = Iy
		b = (T - I).reshape(-1)
		delta_p = np.linalg.inv(A.T @ A) @ np.matmul(A.T, b)
		p = p + delta_p
		mag = np.sum(delta_p ** 2)
		if mag < 1e-5 :
			break
	return p
