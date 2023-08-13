import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
def queue_worker(q):
	while True:
		id, ind_list, feat_list, filename, dictionary, layer_num, K = q.get()
		ind_list.append(id)
		image = imageio.imread(filename)
		wordmap = visual_words.get_visual_words(image, dictionary)
		feat_list.append(get_feature_from_wordmap_SPM(wordmap, layer_num, K))
		#if id % 100 == 0:
			#print("doing "+str(id)+"th task...")
		q.task_done()

def build_recognition_system(num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,M)
	* labels: numpy.ndarray of shape (N)
	* dictionary: numpy.ndarray of shape (K,3F)
	* SPM_layer_num: number of spatial pyramid layers
	'''
	train_data = np.load("../data/train_data.npz", allow_pickle = True)
	dictionary = np.load("dictionary.npy")
	N = train_data['labels'].shape[0]
	rand_features = []
	ind = []
	layer_num = 3
	q = queue.Queue(maxsize=0)
	for i in range(num_workers):
		worker = threading.Thread(target = queue_worker, args=(q,))
		worker.setDaemon(True)
		worker.start()
	for i in range(N):
		name = train_data['image_names'][i][0]
		name = "../data/" + name
		q.put((i, ind, rand_features, name, dictionary, layer_num, dictionary.shape[0]))
	q.join()
	features = np.zeros((N, int(dictionary.shape[0] * ((4 ** layer_num) - 1) / 3)))
	for i in range(N):
		features[ind[i]] = rand_features[i]
	np.savez("trained_system.npz", dictionary=dictionary, features = features, labels = train_data['labels'], SPM_layer_num = layer_num)

def evaluate_recognition_system(num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	trained_system = np.load("trained_system.npz", allow_pickle=True)
	dict = trained_system['dictionary']
	features = trained_system['features']
	train_labels = trained_system['labels']
	layer_num = trained_system['SPM_layer_num']
	N = test_data['labels'].shape[0]
	test_labels = test_data['labels']
	q = queue.Queue(maxsize=0)
	rand_features = []
	ind = []
	for i in range(num_workers):
		worker = threading.Thread(target = queue_worker, args=(q,))
		worker.setDaemon(True)
		worker.start()
	
	for i in range(N): 
		name = test_data['image_names'][i][0]
		name = "../data/" + name
		q.put((i, ind, rand_features, name, dict, layer_num, dict.shape[0]))
	q.join()
	C_map = np.zeros((8, 8))
	for i in range(N):
		dist = distance_to_set(rand_features[i], features)
		C_map[test_labels[ind[i]]][train_labels[np.argmax(dist)]] = C_map[test_labels[ind[i]]][train_labels[np.argmax(dist)]] + 1
		'''if test_labels[ind[i]] != train_labels[np.argmax(dist)]:
			if np.sum(dist == np.min(dist)) != 1:
				name = test_data['image_names'][ind[i]][0]
				print(name)
				potent = train_labels[dist == np.min(dist)]
				print(potent)
		'''
	acc = np.trace(C_map) / np.sum(C_map)
	return C_map, acc




def get_image_feature(file_path,dictionary,layer_num,K):
	'''
	Extracts the spatial pyramid matching feature.

	[input]
	* file_path: path of image file to read
	* dictionary: numpy.ndarray of shape (K,3F)
	* layer_num: number of spatial pyramid layers
	* K: number of clusters for the word maps

	[output]
	* feature: numpy.ndarray of shape (K)
	'''
	image = imageio.imread(file_path)
	wordmap = visual_words.get_visual_words(image,dictionary)
	SPM_feature = get_feature_from_wordmap_SPM(wordmap, layer_num, K)
	return SPM_feature


def distance_to_set(word_hist,histograms):
	'''
	Compute similarity between a histogram of visual words with all training image histograms.

	[input]
	* word_hist: numpy.ndarray of shape (K)
	* histograms: numpy.ndarray of shape (N,K)

	[output]
	* sim: numpy.ndarray of shape (N)
	'''
	sim = np.minimum(word_hist, histograms)
	sim = np.sum(sim, axis = 1)
	return sim

def get_feature_from_wordmap(wordmap,dict_size):
	'''
	Compute histogram of visual words.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* dict_size: dictionary size K

	[output]
	* hist: numpy.ndarray of shape (K)
	'''
	wordmap = wordmap * dict_size
	wordmap = wordmap.astype(np.int32)
	hist, edges = np.histogram(wordmap, bins = dict_size, range = (0, dict_size))
	hist = np.array(hist)
	hist = hist / np.sum(hist)
	#import matplotlib.pyplot as plt
	#plt.plot(edges[:-1], hist)
	#plt.show()
	#print(edges)
	return hist

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
	'''
	Compute histogram of visual words using spatial pyramid matching.

	[input]
	* wordmap: numpy.ndarray of shape (H,W)
	* layer_num: number of spatial pyramid layers
	* dict_size: dictionary size K
	[output]
	* hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3)
	'''
	hist_all = np.zeros((dict_size, int((4 ** layer_num - 1) / 3)))
	#print("hist_all shape:", hist_all.shape)
	H, W = wordmap.shape
	L = layer_num - 1
	cur_hist_size = (4 ** (L + 1) - 1) / 3
	d_h = (H - 1) // (2 ** L) + 1
	d_w = (W - 1) // (2 ** L) + 1
	hist_L = np.zeros((dict_size, 2 ** L, 2 ** L))
	for h in range(2 ** L):
		for w in range(2 ** L):
			tempmap = wordmap[h * d_h : (h + 1) * d_h, w * d_w : (w + 1) * d_w]
			hist_L[:,h,w] = get_feature_from_wordmap(tempmap, dict_size)
			hist_L[:,h,w] = hist_L[:,h,w] * tempmap.shape[0] * tempmap.shape[1]
	#print("hist_L's shape:", hist_L.shape)
	cur_hist_size = int(cur_hist_size)
	hist_all[:,cur_hist_size - int(4 ** L):cur_hist_size] = hist_L.reshape(dict_size, -1) * 0.5
	cur_hist_size = cur_hist_size - 4 ** L
	cur_hist_size = int(cur_hist_size)
	for l in range(L - 1, -1, -1):
		hist_l = np.zeros((dict_size, 2 ** l, 2 ** l))
		for h in range(2 ** l):
			for w in range(2 ** l):
				hist_l[:,h,w] = hist_L[:,h*2,w*2] + hist_L[:,h*2,w*2+1] + hist_L[:,h*2+1,w*2] + hist_L[:,h*2+1,w*2+1]
		hist_L = hist_l
		if l != 0:
			hist_all[:,cur_hist_size - int(4 ** l):cur_hist_size] = hist_L.reshape(dict_size, -1) * (2.0 ** (l-L-1))
		else :
			hist_all[:,cur_hist_size - int(4 ** l):cur_hist_size] = hist_L.reshape(dict_size, -1) * (2.0 ** (-L))
		cur_hist_size = cur_hist_size - 4 ** l
		cur_hist_size = int(cur_hist_size)
	hist_all = hist_all.reshape(-1)
	hist_all = hist_all / np.sum(hist_all)
	return hist_all