import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import matplotlib.pyplot as plt
import util
import random

def extract_filter_responses(image):
	'''
	Extracts the filter responses for the given image.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	[output]
	* filter_responses: numpy.ndarray of shape (H,W,3F)
	'''
	im_shp = image.shape
	if len(im_shp) == 2:
		image = np.stack((image, image, image), axis = -1)
	elif im_shp[-1] == 4:
		image = skimage.color.rgba2rgb(image)
	if image.max() > 10:
		image = np.float32(image) / 255
	if image.min() < 0:
		image = (image + 1) / 2
	#remember to check all the image data type
	image = skimage.color.rgb2lab(image)
	filter_responses = []
	scales = [1, 2, 4, 8, 8 * 2 ** 0.5]
	scales = np.array(scales, dtype = np.float32)
	for scale in scales : 
		for chn in range(3):
			filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,chn], sigma = scale)) # gaussian
		for chn in range(3):
			filter_responses.append(scipy.ndimage.gaussian_laplace(image[:,:,chn], sigma = scale)) # laplace of gaussian
		for chn in range(3):
			filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,chn], sigma = scale, order = (0, 1))) # gaussian of y derivative
		for chn in range(3):
			filter_responses.append(scipy.ndimage.gaussian_filter(image[:,:,chn], sigma = scale, order = (1, 0))) # gaussian of x detivative

	filter_responses = np.stack(filter_responses, axis = -1)
	return filter_responses






def get_visual_words(image,dictionary):
	'''
	Compute visual words mapping for the given image using the dictionary of visual words.

	[input]
	* image: numpy.ndarray of shape (H,W) or (H,W,3)
	
	[output]
	* wordmap: numpy.ndarray of shape (H,W)
	'''
	filter_responses = extract_filter_responses(image)
	H = image.shape[0]
	W = image.shape[1]
	filter_responses = filter_responses.reshape((H * W, -1))
	dist_map = scipy.spatial.distance.cdist(filter_responses, dictionary, metric = 'euclidean')
	dist_map = np.argmin(dist_map, axis = 1)
	wordmap = dist_map.reshape((H, W)).astype(np.float32) / dictionary.shape[0]
	#wordmap = (wordmap - wordmap.min()) / (wordmap.max() - wordmap.min())
	return wordmap


def compute_dictionary_one_image(args):
	'''
	Extracts random samples of the dictionary entries from an image.
	This is a function run by a subprocess.

	[input]
	* i: index of training image
	* alpha: number of random samples
	* image_path: path of image file
	* time_start: time stamp of start time

	[saved]
	* sampled_response: numpy.ndarray of shape (alpha,3F)
	'''
	i,alpha,image_path, _ = args
	#if i % 200 == 0: print(_)
	image = skimage.io.imread(image_path)
	filter_response = extract_filter_responses(image)
	F = 20
	filter_response = filter_response.reshape((-1,3*F))
	pick = np.random.choice(filter_response.shape[0], alpha)
	filter_response = filter_response[pick, :]
	save_path = "../temp/" + str(i)
	np.save(save_path, filter_response)
	#print("success:"+str(i))
	return 


def compute_dictionary(num_workers=2):
	'''
	Creates the dictionary of visual words by clustering using k-means.

	[input]
	* num_workers: number of workers to process in parallel
	
	[saved]
	* dictionary: numpy.ndarray of shape (K,3F)
	'''

	train_data = np.load("../data/train_data.npz", allow_pickle = True)
	N = train_data['labels'].shape[0]
	alpha = 50
	K = 100
	F = 20
	pool = multiprocessing.Pool(processes = num_workers)

	for i in range(N): 
		name = train_data['image_names'][i][0]
		name = "../data/" + name
		arg = (i, alpha, name, time.time())
		pool.apply_async(compute_dictionary_one_image, (arg,))
		#compute_dictionary_one_image(arg)
		#print(i)
		#if i % 200 == 0:
			#print("loading " + str(i)+"th task...")
	#print(">>>finish loading")
	pool.close()
	pool.join()
	#print(">>>finish tasks")
	filter_responses = np.zeros((N, alpha, 3 * F))
	for i in range(N):
		a = np.load('../temp/'+str(i)+'.npy')
		filter_responses[i, :, :] = a
	filter_responses = filter_responses.reshape(-1, 3 * F)
	#print("doing kmeans...")
	kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(filter_responses)
	dictionary = kmeans.cluster_centers_
	#print(">>>finish kmeans")
	np.save("dictionary.npy", dictionary)
	#print(">>>finish saving dictionary")
	