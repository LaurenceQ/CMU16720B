import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade
# write your script here, we recommend the above libraries for making your animation

frames = np.load("../data/carseq.npy")
ims = []
#N = 26
#frames = frames[:,:,:N]
fig = plt.figure()
bboxes = np.zeros((frames.shape[-1], 4))
save_bboxes = np.zeros((frames.shape[-1], 4))
bbox = np.array([59, 116, 145, 151], dtype = np.float64)
temp_box = np.array([59, 116, 145, 151], dtype = np.float64)
last = 0
Ps = np.zeros((frames.shape[-1], 2))
for t in range(frames.shape[-1]):
    if t > 0:
        dp = LucasKanade(frames[:,:,t-1], frames[:,:,t], bbox, p0 = np.array((0, 0)))
        bbox[0::2] += dp[0]
        bbox[1::2] += dp[1]
        temp_box[:] = bboxes[0]
        temp_box[0::2] += Ps[last][0]
        temp_box[1::2] += Ps[last][1]
        Ps[t,:] = LucasKanade(frames[:,:,last], frames[:,:,t], temp_box, p0 = Ps[t-1] - Ps[last]) + Ps[last]
        Ps_star = LucasKanade(frames[:,:,0], frames[:,:,t], bboxes[0], Ps[t])
        if np.sum((Ps_star - Ps[t]) ** 2) < 1e-3 : 
            last = t
    bboxes[t,:] = bbox
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
    #print(bboxes[0].shape, Ps[t].shape)
    rect_correct = patches.Rectangle((bboxes[0][0] + Ps[t][0], bboxes[0][1] + Ps[t][1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='g',facecolor='none')
    save_bboxes[t,:] = bboxes[0]
    save_bboxes[t,0::2] += Ps[t][0]
    save_bboxes[t,1::2] += Ps[t][1]
    ims.append([ plt.imshow(frames[:,:,t], cmap = 'gray', animated = True),
    plt.gca().add_patch(rect),
    plt.gca().add_patch(rect_correct),
    ])
    #print(t)
np.save("carseqrects-wcrt", save_bboxes)
print(save_bboxes[:30, :])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                               repeat_delay=1000)
plt.show()
num = 1
fig, axes = plt.subplots(nrows = 1, ncols = 5)
for t in range(frames.shape[-1]):
    if t in [1, 100, 200, 300, 400]:
        bbox = bboxes[t]
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='r',facecolor='none')
        rect_correct = patches.Rectangle((bboxes[0][0] + Ps[t][0], bboxes[0][1] + Ps[t][1]), bbox[2] - bbox[0], bbox[3] - bbox[1],linewidth=1,edgecolor='g',facecolor='none')
        axes[num-1].imshow(frames[:,:,t], cmap = 'gray')
        axes[num-1].add_patch(rect)
        axes[num-1].add_patch(rect_correct)
        num += 1

plt.show()