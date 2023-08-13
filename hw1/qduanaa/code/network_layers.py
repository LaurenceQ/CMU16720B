import numpy as np
import scipy.ndimage
import os,time
import skimage.transform
def extract_deep_feature(x,vgg16_weights):
	'''
	Extracts deep features from the given VGG-16 weights.

	[input]
	* x: numpy.ndarray of shape (H,W,3)
	* vgg16_weights: numpy.ndarray of shape (L,3)

	[output]
	* feat: numpy.ndarray of shape (K)
	'''
	name_weight = vgg16_weights
	L = len(name_weight)
	image = x
	if image.max() > 10:
		image = np.float32(image) / 255
	if image.min() < 0:
		image = (image + 1) / 2
	image = skimage.transform.resize(image, (224, 224, 3))
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	image = (image - mean) / std
	
	cnt_linear = 0
	for l in name_weight:
		if l[0] == 'conv2d':
			image = multichannel_conv2d(image, l[1], l[2])
		elif l[0] == 'relu':
			image = relu(image)
		elif l[0] == 'maxpool2d':
			image = max_pool2d(image, l[1])
		elif l[0] == 'linear':
			if cnt_linear == 0:
				image = image.transpose((2, 0, 1))
				image = image.reshape((-1,))
			image = linear(image, l[1], l[2])
			cnt_linear = cnt_linear + 1
			if cnt_linear == 2:
				break

	return image


def multichannel_conv2d(x,weight,bias):
	'''
	Performs multi-channel 2D convolution.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim,kernel_size,kernel_size)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* feat: numpy.ndarray of shape (H,W,output_dim)
	'''
	result_x = np.zeros((x.shape[0], x.shape[1], weight.shape[0]))
	for i in range(weight.shape[0]):
		for j in range(x.shape[2]):
			temp = scipy.ndimage.convolve(x[:,:,j], weight[i,j,:,:])
			result_x[:,:,i] += temp
		result_x[:,:,i] = result_x[:,:,i] + bias[i]
	return result_x



def relu(x):
	'''
	Rectified linear unit.

	[input]
	* x: numpy.ndarray

	[output]
	* y: numpy.ndarray
	'''
	return np.maximum(x, 0)

def max_pool2d(x,size):
	'''
	2D max pooling operation.

	[input]
	* x: numpy.ndarray of shape (H,W,input_dim)
	* size: pooling receptive field

	[output]
	* y: numpy.ndarray of shape (H/size,W/size,input_dim)
	'''
	N = x.shape[0]
	k_size = size
	temp = np.zeros(((N - 1) // k_size + 1, x.shape[1], x.shape[2]))
	for i in range(0, N, k_size):
		temp[i//k_size,:,:] = np.max(x[i:i+k_size,:,:], axis = 0)
	M = x.shape[1]
	tempp = np.zeros((temp.shape[0],  (M - 1) // k_size + 1, x.shape[2]))
	for i in range(0, M, k_size):
		tempp[:,i//k_size,:] = np.max(temp[:,i:i+k_size,:], axis = 1)
	return tempp


def linear(x,W,b):
	'''
	Fully-connected layer.

	[input]
	* x: numpy.ndarray of shape (input_dim)
	* weight: numpy.ndarray of shape (output_dim,input_dim)
	* bias: numpy.ndarray of shape (output_dim)

	[output]
	* y: numpy.ndarray of shape (output_dim)
	'''
	return np.matmul(W, x) + b


