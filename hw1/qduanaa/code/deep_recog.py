import numpy as np
import multiprocessing
import threading
import queue
import imageio
import os,time
import torch
import skimage.transform
import torchvision.transforms
import util
import network_layers
import scipy

def build_recognition_system(vgg16,num_workers=2):
	'''
	Creates a trained recognition system by generating training features from all training images.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[saved]
	* features: numpy.ndarray of shape (N,K)
	* labels: numpy.ndarray of shape (N)
	'''


	train_data = np.load("../data/train_data.npz", allow_pickle = True)
	N = train_data['labels'].shape[0]
	my_vgg = vgg16
	my_vgg.classifier = torch.nn.Sequential(*(list(vgg16.classifier.children())[:-3]))
	#cuda = torch.device("cuda:0")
	#my_vgg = my_vgg.to(cuda)
	for param in my_vgg.parameters():
		param.requires_grad = False
	features = []
	for i in range(N):
		name = train_data['image_names'][i][0]
		name = "../data/" + name
		features.append(get_image_feature((i, name, my_vgg)).numpy())
		#if i % 100 == 0:
			#print("training "+ str(i)+"th data...")
	features = np.squeeze(np.array(features))
	np.savez("trained_system_deep.npz", features = features, labels = train_data['labels'])
	#print(">>>Finish")



def evaluate_recognition_system(vgg16,num_workers=2):
	'''
	Evaluates the recognition system for all test images and returns the confusion matrix.

	[input]
	* vgg16: prebuilt VGG-16 network.
	* num_workers: number of workers to process in parallel

	[output]
	* conf: numpy.ndarray of shape (8,8)
	* accuracy: accuracy of the evaluated system
	'''

	test_data = np.load("../data/test_data.npz", allow_pickle=True)
	trained_system = np.load("trained_system_deep.npz", allow_pickle=True)
	features = trained_system['features']
	train_labels = trained_system['labels']
	N = test_data['labels'].shape[0]
	test_labels = test_data['labels']
	test_features = []
	my_vgg = vgg16
	my_vgg.classifier = torch.nn.Sequential(*(list(vgg16.classifier.children())[:-3]))
	#cuda = torch.device("cuda:0")
	#my_vgg = my_vgg.to(cuda)
	for param in my_vgg.parameters():
		param.requires_grad = False

	for i in range(N): 
		name = test_data['image_names'][i][0]
		name = "../data/" + name
		test_features.append(get_image_feature((i, name, my_vgg)).numpy())
		#if i % 100 == 0:
			#print("testing "+ str(i)+"th data...")
	C_map = np.zeros((8, 8))
	for i in range(N):
		dist = distance_to_set(np.squeeze(test_features[i]), features)
		C_map[test_labels[i]][train_labels[np.argmax(dist)]] = C_map[test_labels[i]][train_labels[np.argmax(dist)]] + 1
		'''if test_labels[i] != train_labels[np.argmax(dist)]:
			if np.sum(dist == np.min(dist)) != 1:
				name = test_data['image_names'][i][0]
				print(name)
				potent = train_labels[dist == np.min(dist)]
				print(potent)
		'''
	acc = np.trace(C_map) / np.sum(C_map)
	#print(">>>Finish")
	return C_map



def preprocess_image(image):
	'''
	Preprocesses the image to load into the prebuilt network.

	[input]
	* image: numpy.ndarray of shape (H,W,3)

	[output]
	* image_processed: torch.Tensor of shape (3,H,W)
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
	image = skimage.transform.resize(image, (224, 224))
	mean = [0.485, 0.456, 0.406]
	std = [0.229, 0.224, 0.225]
	image = (image - mean) / std
	image = np.transpose(image, (2, 0, 1))
	#cuda = torch.device("cuda:0")
	image = torch.from_numpy(image)
	#image = image.to(cuda)
	return image

def get_image_feature(args):
	'''
	Extracts deep features from the prebuilt VGG-16 network.
	This is a function run by a subprocess.
 	[input]
	* i: index of training image
	* image_path: path of image file
	* vgg16: prebuilt VGG-16 network.
	* time_start: time stamp of start time
 	[saved]
	* feat: evaluated deep feature
	'''

	i,image_path,vgg16 = args
	image = imageio.imread(image_path)
	image = preprocess_image(image)
	image = image.unsqueeze(0)
	return vgg16(image.double())

def distance_to_set(feature,train_features):
	'''
	Compute distance between a deep feature with all training image deep features.

	[input]
	* feature: numpy.ndarray of shape (K)
	* train_features: numpy.ndarray of shape (N,K)

	[output]
	* dist: numpy.ndarray of shape (N)
	'''
	sim = -np.sum(np.square(train_features - feature), axis = 1)
	return sim