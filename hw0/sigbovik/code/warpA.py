import numpy as np
def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""
    print(im.shape)
    print(output_shape)
    A = np.linalg.inv(A)
    org_x = np.zeros(output_shape)
    org_y = np.zeros(output_shape)
    cord_x = np.arange(output_shape[0])
    cord_y = np.arange(output_shape[1]) 
    temp_x, temp_y = np.meshgrid(cord_y * A[0][1], cord_x * A[0][0])
    org_x = temp_x + temp_y + A[0][2]
    temp_x, temp_y = np.meshgrid(cord_y * A[1][1], cord_x * A[1][0])
    org_y = temp_x + temp_y + A[1][2]
    org_x = org_x.round()
    org_y = org_y.round()
    org_x = org_x.astype(np.int16)
    org_y = org_y.astype(np.int16)
    org_x[org_x >= im.shape[0]] = -1
    org_y[org_y >= im.shape[1]] = -1
    mask = np.logical_and((org_x >= 0), (org_y >= 0))
    output = np.where(mask, im[org_x, org_y], 0)
    return output
