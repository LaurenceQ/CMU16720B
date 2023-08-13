import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

frames = np.load("../data/carseq.npy")
bbox = np.array([59, 116, 145, 151], dtype = np.float64)
ims = []
#N = 50
#frames = frames[:,:,:N]
fig = plt.figure()
bboxes = np.zeros((frames.shape[-1], 4))
for t in range(frames.shape[-1]):
    if t > 0:
        dp = LucasKanade(frames[:,:,t-1], frames[:,:,t], bbox, p0 = np.array((0, 0)))
        #print(t, dp)
        bbox[0::2] += dp[0]
        bbox[1::2] += dp[1]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
    bboxes[t,:] = bbox
    ims.append([ plt.imshow(frames[:,:,t], cmap = 'gray', animated = True),
    plt.gca().add_patch(rect) ])
np.save("carseqrects", bboxes)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                               repeat_delay=1000)
plt.show()
num = 1
fig, axes = plt.subplots(nrows = 1, ncols = 5)
for t in range(frames.shape[-1]):
    if t in [1, 100, 200, 300, 400]:
        bbox = bboxes[t]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
        axes[num-1].imshow(frames[:,:,t], cmap = 'gray')
        axes[num-1].add_patch(rect)
        num += 1

plt.show()