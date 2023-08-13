import numpy as np
def cir_shift(dx, dy, tar):
    cur = np.zeros(tar.shape)
    tmp = np.zeros(tar.shape)
    X = tar.shape[0]
    Y = tar.shape[1]
    if dx < 0 :
        cur[:dx,:] = tar[-dx:,:]
        cur[dx:,:] = tar[:-dx,:]
    else :
        cur[dx:,:] = tar[:X-dx,:]
        cur[:dx,:] = tar[X-dx:,:]
    if dy < 0:
        tmp[:, :dy] = cur[:, -dy:]
        tmp[:, dy:] = cur[:, :-dy]
    else :
        tmp[:, dy:] = cur[:, :Y-dy]
        tmp[:, :dy] = cur[:, Y-dy:]
    return tmp 
 
def circShift(dx, dy, A):
        shift_output = np.roll(A, (dx, dy), axis=(0, 1))
        return shift_output
def duizhun(base, tar):
    min_loss = 1e18
    ans = np.zeros(base.shape)
    for dx in range(-30, 31):
        for dy in range(-30, 31):
            cur = cir_shift(dx, dy, tar)
            loss = np.sum(np.square(base-cur))
            if loss < min_loss:
                min_loss = loss
                ans = cur 
    return ans
 
def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""
    ans_blue = duizhun(red, blue)
    ans_green = duizhun(red, green)
    return np.stack((red, ans_green, ans_blue), axis = 2)
