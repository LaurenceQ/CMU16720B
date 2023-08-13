import numpy as np
import cv2
import matplotlib.pyplot as plt
from BRIEF import briefLite
from BRIEF import briefMatch
im1 = cv2.imread('../data/model_chickenbroth.jpg')
if len(im1.shape)==3:
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
if im1.max()>10:
    im1 = np.float32(im1)/255
locs1, desc1 = briefLite(im1)
COL = im1.shape[1]
ROW = im1.shape[0]
matched_num = []
print(locs1.shape[0])
for deg in range(0, 360, 10):
    rotate_mat = cv2.getRotationMatrix2D((COL // 2, ROW // 2), deg, scale = 1)
    im2 = cv2.warpAffine(im1, rotate_mat, (im1.shape[1], im1.shape[0]))
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    matched_num.append(matches.shape[0])

matched_num = np.array(matched_num)
degs = range(0, 36)
fig, ax = plt.subplots(constrained_layout = True)
rect = ax.bar(degs, matched_num)
ax.set_title(" Rotation angle VS The number of correct matches")
ax.set_ylabel("Number of correct matches")
ax.set_xlabel("Rotation angle/10 (degrees)")
plt.show()