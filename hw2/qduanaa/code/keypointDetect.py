import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    L = len(levels)
    for l in range(1, L):
        DoG_pyramid.append(gaussian_pyramid[:,:,l] - gaussian_pyramid[:,:,l-1])
    DoG_levels = levels[1:]
    DoG_pyramid = np.stack(DoG_pyramid, axis = -1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ##################
    # TO DO ...
    # Compute principal curvature here
    Dxx = cv2.Sobel(DoG_pyramid, ddepth=-1, dx = 2, dy = 0, ksize = 3)
    Dyy = cv2.Sobel(DoG_pyramid, ddepth=-1, dx = 0, dy = 2, ksize = 3)
    Dxy = cv2.Sobel(DoG_pyramid, ddepth=-1, dx = 1, dy = 1, ksize = 3)
    denominator = Dxx * Dyy - (Dxy ** 2)
    denominator = np.where(denominator != 0, denominator, 1e-9)
    principal_curvature = ((Dxx + Dyy) ** 2) / denominator
    return principal_curvature

def roll_and_compare(func, vec, org_im):
    return func(org_im, np.roll(org_im, shift = vec, axis = (0, 1))) == org_im

def compare(func, org, tar):
    return func(org, tar) == org
def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    L = DoG_pyramid.shape[-1]
    Row = DoG_pyramid.shape[0]
    Col = DoG_pyramid.shape[1]
    max_map = np.ones((L, Row, Col))
    min_map = np.ones((L, Row, Col))
    for l in range(L):
        cur_max_map = np.ones(DoG_pyramid.shape[:-1])
        cur_min_map = np.ones(DoG_pyramid.shape[:-1])
        cur_max_map[0,:] = cur_max_map[-1,:] = 0
        cur_max_map[:,0] = cur_max_map[:,-1] = 0
        cur_min_map[0,:] = cur_min_map[-1,:] = 0
        cur_min_map[:,0] = cur_min_map[:,-1] = 0
        cur_pyr = DoG_pyramid[:,:,l]
        for vec in direction :
            cur_max_map = np.logical_and(cur_max_map, roll_and_compare(np.maximum, vec, cur_pyr))
            cur_min_map = np.logical_and(cur_min_map, roll_and_compare(np.minimum, vec, cur_pyr))
        max_map[l] = cur_max_map
        min_map[l] = cur_min_map
    
    locsDoG = []
    for l in range(0, L):
        cur_pyr = DoG_pyramid[:,:,l]
        if l == 0:
            nex_pyr = DoG_pyramid[:,:,l+1]
            res_max_map = np.logical_and(max_map[l], compare(np.maximum, cur_pyr, nex_pyr))
            res_min_map = np.logical_and(min_map[l], compare(np.minimum, cur_pyr, nex_pyr))
        elif l == L - 1 : 
            las_pyr = DoG_pyramid[:,:,l-1]
            res_max_map = np.logical_and(max_map[l], compare(np.maximum, cur_pyr, las_pyr))
            res_min_map = np.logical_and(min_map[l], compare(np.minimum, cur_pyr, las_pyr))
        else :
            nex_pyr = DoG_pyramid[:,:,l+1]
            las_pyr = DoG_pyramid[:,:,l-1]
            res_max_map = np.logical_and(max_map[l], compare(np.maximum, cur_pyr, nex_pyr))
            res_min_map = np.logical_and(min_map[l], compare(np.minimum, cur_pyr, nex_pyr))
            res_max_map = np.logical_and(max_map[l], compare(np.maximum, cur_pyr, las_pyr))
            res_min_map = np.logical_and(min_map[l], compare(np.minimum, cur_pyr, las_pyr))
        cur_magnitude = np.absolute(DoG_pyramid[:,:,l])
        cur_curvature = principal_curvature[:,:,l]
        thresh_map = np.logical_and(cur_pyr > th_contrast, cur_curvature < th_r)
        res_max_map = np.logical_and(res_max_map, thresh_map)
        res_min_map = np.logical_and(res_min_map, thresh_map)
        num_max = np.sum(res_max_map)
        num_min = np.sum(res_min_map)
        '''
        print(np.sum(thresh_map))
        print(np.sum(res_max_map))
        print(np.sum(res_min_map))
        print(np.where(res_max_map == True))
        '''
        num_max = np.sum(res_max_map)
        num_min = np.sum(res_min_map)
        if num_max : 
            tmp = list(np.where(res_max_map == True))[::-1]
            tmp.append(l * np.ones(num_max))
            tmp = np.stack(tmp, axis = -1)
            locsDoG.append(tmp)
        if num_min:
            tmp = list(np.where(res_min_map == True))[::-1]
            tmp.append(l * np.ones(num_min))
            locsDoG.append(np.stack(tmp, axis = -1))

    ##############
    #  TO DO ...
    # Compute locsDoG here
    locsDoG = np.concatenate(locsDoG, axis = 0)
    locsDoG = locsDoG.astype(np.int16)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    return locsDoG, gauss_pyramid







if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    #im = cv2.imread('../data/model_chickenbroth.jpg')
    im = cv2.imread('../data/chickenbroth_01.jpg')
    #im =cv2.imread('../data/pf_scan_scaled.jpg')
    #cv2.imshow('chicken.jpg', im)
    #print(im.shape)
    im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)

    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    #print(locsDoG)
    for coord in locsDoG:
        im = cv2.circle(im, coord[:-1], 1, (0,255,0))
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', im)
    im = np.round(im * 255)
    cv2.imwrite('../results/Q1.5.png', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()
    #print("XBX")
    #print(locsDoG)
    # test DoG detector
    #locsDoG, gaussian_pyramid = DoGdetector(im)
    #print(locsDoG.shape)