import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

if __name__ == '__main__':

	#num_cores = util.get_num_CPU()
	num_cores = 6
	print(num_cores)
	#path_img = "../data/windmill/sun_bcyrzigcapliwiox.jpg"
	#path_img = "../data/highway/sun_aagkjhignpmigxkv.jpg"
	#path_img = "../data/highway/sun_bdjkcfnppbworqx.jpg"
	path_img = "../data/highway/sun_byinrjsphbeujidj.jpg"
	path_img = "../data/baseball_field/sun_bmcvqayffuzocbfi.jpg"
	image = skimage.io.imread(path_img)
	skimage.io.imsave("Lala.jpg", image)
	image = image.astype('float')/255
	#filter_responses = visual_words.extract_filter_responses(image)
	#util.display_filter_responses(filter_responses)
	#num_cores = 
	#visual_words.compute_dictionary(num_workers=num_cores)
	
	dictionary = np.load('dictionary.npy')
	#print("dictionary shape:", dictionary.shape)
	#img = visual_words.get_visual_words(image,dictionary)
	#filename = "example.jpg"
	#util.save_wordmap(img, filename)
	#print(">>>Finishing visualizing")
	#print(img)
	#hist = visual_recog.get_feature_from_wordmap(img, dictionary.shape[0])
	#visual_recog.build_recognition_system(num_workers=num_cores)

	conf, acc = visual_recog.evaluate_recognition_system(num_workers=num_cores)
	print(conf)
	print(acc)

	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#vgg16.eval()
	#deep_recog.build_recognition_system(vgg16,num_workers=num_cores)
	#vgg16 = torchvision.models.vgg16(pretrained=True).double()
	#vgg16.eval()
	#conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores)
	#print(conf)
	#print(np.diag(conf).sum()/conf.sum())

