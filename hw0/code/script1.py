from alignChannels import alignChannels
import numpy as np
from PIL import Image as im
# Problem 1: Image Alignment

# 1. Load images (all 3 channels)
red = np.load(r'../data/red.npy')
green = np.load(r'../data/green.npy')
blue = np.load(r'../data/blue.npy')
# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)
rgbResult = rgbResult.astype(np.uint8)

data = im.fromarray(rgbResult)
data.save('../results/rgb_output.jpg')
# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)
