import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanadeBasis import LucasKanadeBasis
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

frames = np.load("../data/sylvseq.npy")
base = np.load("../data/sylvbases.npy")
save_bbox = np.array([101, 61, 155, 107], dtype = np.float64)
bbox = np.array([101, 61, 155, 107], dtype = np.float64)

ims = []
#N = 50
#frames = frames[:,:,:N]
fig = plt.figure()
save_bboxes = np.zeros((frames.shape[-1], 4))
bboxes = np.zeros((frames.shape[-1], 4))
Ps = np.zeros((frames.shape[-1], 2))
last = 0
for t in range(frames.shape[-1]):
    if t > 0:
        dp = LucasKanadeBasis(frames[:,:,t-1], frames[:,:,t], save_bbox, base)
        save_bbox[0::2] += dp[0]
        save_bbox[1::2] += dp[1]
        bbox[:] = bboxes[0]
        bbox[0::2] += Ps[last][0]
        bbox[1::2] += Ps[last][1]
        Ps[t,:] = LucasKanade(frames[:,:,last], frames[:,:,t], bbox, p0 = Ps[t-1] - Ps[last]) + Ps[last]
        Ps_star = LucasKanade(frames[:,:,0], frames[:,:,t], bboxes[0], Ps[t])
        if np.sum((Ps_star - Ps[t]) ** 2) < 1e-3 : 
            last = t
    save_bboxes[t,:] = save_bbox
    bboxes[t,:] = save_bboxes[0]
    bboxes[t,0::2] += Ps[t][0]
    bboxes[t,1::2] += Ps[t][1]
    bbox[:] = bboxes[t]
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
    rect_save = patches.Rectangle((save_bbox[0], save_bbox[1]), save_bbox[2] - save_bbox[0], save_bbox[3] - save_bbox[1],linewidth=1,edgecolor='g',facecolor='none')
    ims.append([ plt.imshow(frames[:,:,t], cmap = 'gray', animated = True),
    plt.gca().add_patch(rect),
    plt.gca().add_patch(rect_save),
    ])
    print(t)
np.save("sylvseqrects", save_bboxes)
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                               repeat_delay=1000)
plt.show()
num = 1
fig, axes = plt.subplots(nrows = 1, ncols = 5)
for t in range(frames.shape[-1]):
    if t in [1, 200, 300, 350, 400]:
        bbox = bboxes[t]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
        save_bbox = save_bboxes[t]
        save_rect = patches.Rectangle((save_bbox[0], save_bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='g',facecolor='none')
        axes[num-1].imshow(frames[:,:,t], cmap = 'gray')
        axes[num-1].add_patch(rect)
        axes[num-1].add_patch(save_rect)
        num += 1

plt.show()