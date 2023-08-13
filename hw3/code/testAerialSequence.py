import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from SubtractDominantMotion import SubtractDominantMotion
# write your script here, we recommend the above libraries for making your animation
frames = np.load("../data/aerialseq.npy")
ims = []
#N = 13
#frames = frames[:,:,:N]
fig = plt.figure()
for t in range(1, frames.shape[-1]):
    if t > 0:
        mask = SubtractDominantMotion(frames[:,:,t-1], frames[:,:,t])
        masked = np.ma.masked_where(mask == False, mask)
        ims.append([ plt.imshow(frames[:,:,t - 1], cmap = 'gray', animated = True),
        plt.imshow(masked, cmap = 'hsv', interpolation = 'none', alpha = 0.4) ])
    #if t % 10 == 0:
    print(t)
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                               repeat_delay=2000)
plt.show()
num = 1
fig, axes = plt.subplots(nrows = 1, ncols = 4)
for t in range(frames.shape[-1]):
    if t in [30, 60, 90, 120]:
        mask = SubtractDominantMotion(frames[:,:,t-1], frames[:,:,t])
        masked = np.ma.masked_where(mask == False, mask)
        axes[num-1].imshow(frames[:,:,t - 1], cmap = 'gray', animated = True),
        axes[num-1].imshow(masked, cmap = 'hsv', interpolation = 'none', alpha = 0.4)
        axes[num-1].axis('off')
        num += 1

plt.show()