import numpy as np
from LucasKanadeAffine import LucasKanadeAffine 
from InverseCompositionAffine import InverseCompositionAffine
from scipy.ndimage import affine_transform, binary_dilation, binary_erosion
import matplotlib.pyplot as plt
def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1 
	# Output:
	#	mask: [nxm]
    # put your implementation here
	#M = LucasKanadeAffine(image1, image2)
	M = InverseCompositionAffine(image1, image2)
	affine_im = affine_transform(image2, M)
	valid = affine_transform(np.ones(image2.shape), M)

	diff = (image1 - affine_im) ** 2
	diff *= valid
	'''fig, axes = plt.subplots(1, 4)
	print(image1.shape)
	axes[0].set_title('image1')
	axes[0].imshow(image1, cmap = 'gray')
	print(image2.shape)
	axes[1].set_title('image2')
	axes[1].imshow(image2, cmap = 'gray')
	print(affine_im.shape)
	axes[2].set_title('affine_im2')
	axes[2].imshow(affine_im, cmap = 'gray')
	print(diff.shape)
	axes[3].set_title('diff')
	axes[3].imshow(diff, cmap = 'gray')
	plt.show()
	#'''
	mean = diff.mean()
	std = diff.std()
	mask = np.ones(image1.shape, dtype=bool)
	mask = np.logical_and(mask, (diff - mean > std * 1.5))
	mask = binary_dilation(mask, structure = np.ones((5, 5)))
	mask = binary_erosion(mask, structure = np.ones((5, 5)))
	return mask

