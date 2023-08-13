import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    sigma = skimage.restoration.estimate_sigma(image, False, channel_axis = -1)
    image = skimage.restoration.denoise_wavelet(image, sigma, channel_axis = -1)
    gray_image = skimage.color.rgb2gray(image)
    thresh = skimage.filters.threshold_otsu(gray_image)
    bw = (gray_image < thresh).astype(np.float32)
    bw = skimage.morphology.closing(bw, skimage.morphology.square(5))
    labels = skimage.measure.label(bw, connectivity = 2)
    num = labels.max()
    for label in range(1, num + 1):
        label_map = (labels == label)
        y, x = np.where(label_map == True)
        x1 = np.min(x) 
        x2 = np.max(x)
        y1 = np.min(y) 
        y2 = np.max(y) 
        if y2 - y1 < 32 and x2 - x1 < 32 : continue
        bboxes.append([y1, x1, y2, x2])
    temp = []
    for i in range(len(bboxes)):
        contained = False
        y1, x1, y2, x2 = bboxes[i]
        for j in range(len(bboxes)):
            if i == j: continue
            by1, bx1, by2, bx2 = bboxes[j]
            if y1 > by1 and y2 < by2 and x1 > bx1 and x2 < bx2 :
                contained = True
                break
        if not contained :
            temp.append(bboxes[i])
    bw = 1 - bw
    bboxes = temp
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions
    return bboxes, bw