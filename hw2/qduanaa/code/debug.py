import numpy as np
import cv2

im1 = cv2.imread('../data/model_chickenbroth.jpg')
outshape = im1.shape[:-1]
H2to1 = np.zeros([3, 3])
H2to1[0][0] = 3
H2to1[1][1] = 3
H2to1[2][2] = 1
H2to1[0][2] = 0
H2to1[1][2] = 0
pano_im = cv2.warpPerspective(im1, H2to1, (outshape[1] * 3, outshape[0] * 3))
print(H2to1)
cv2.imwrite('../results/panoImg.png', pano_im)
cv2.imshow('panoramas', pano_im)
cv2.waitKey(0)
cv2.destroyAllWindows()

