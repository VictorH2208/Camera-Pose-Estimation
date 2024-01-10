import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.
def dlt_homography(I1pts, I2pts):

    # Create empty palceholder matrix for A
    A = np.empty((0, 9))

    # Update A matrix by creating the A_i for each set of points first following the equation given using np.vstack that stack the A_i vertically
    for i in range(4):
        A_i = np.array([[-I1pts[0, i], -I1pts[1, i], -1, 0, 0, 0, I2pts[0, i] * I1pts[0, i], I2pts[0, i] * I1pts[1, i], I2pts[0, i]],
                       [0, 0, 0, -I1pts[0, i], -I1pts[1, i], -1, I2pts[1, i] * I1pts[0, i], I2pts[1, i] * I1pts[1, i], I2pts[1, i]]])
        A = np.vstack((A, A_i))

    # Find the null space of A to give H because Ah = 0
    h = null_space(A)[:, -1]

    # Reshape h to 3x3 matrix while at the same time dividing by h[-1] to normalize the matrix
    H = np.reshape(h/h[-1], (3,3))

    return H, A

def saddle_point(I):
    # Create empty placeholder for A
    A = np.empty((I.shape[0] * I.shape[1], 6))
    # Fill in A value with x^2 xy y^2 x y 1 where x and y are the calibrated pixel value
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            x =((i+0.5)/I.shape[0]) - 0.5
            y =((j+0.5)/I.shape[1]) - 0.5
            A[i * I.shape[1] + j, :] = np.array([x**2, x*y, y**2, x, y, 1])
    
    # Create empty placeholder for b
    b = np.empty((I.shape[0] * I.shape[1], 1))
    # Fill in b value with intensity from I
    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            b[i * I.shape[1] + j, :] = I[i, j]

    # Solve for linear least squares and extract the coefficients
    coeff = lstsq(A, b)[0]
    alpha = coeff[0]
    beta = coeff[1]
    gamma = coeff[2]
    delta = coeff[3]
    epsilon = coeff[4]
    xi = coeff[5]

    # Find the saddle point
    tmp_1 = -np.linalg.inv(np.array([[2 * alpha, beta], [beta, 2 * gamma]]).reshape(2, 2))
    tmp_2 = np.array([delta, epsilon])
    pt = np.matmul(tmp_1, tmp_2)

    # Adjust the point to be relative to the upper left corner of the patch
    x, y = pt[0] * I.shape[0]  + I.shape[0]//2, pt[1] * I.shape[1] + I.shape[1]// 2
    pt[0], pt[1] = y, x
 
    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---
    Ipts = np.empty((2, Wpts.shape[1]))
    size = Wpts[0,1] - Wpts[0,0]

    # Find the 4 corners of world points
    min_x = np.min(Wpts[0, :]) - size * 1.25
    max_x = np.max(Wpts[0, :]) + size * 1.25
    min_y = np.min(Wpts[1, :]) - size * 1.25
    max_y = np.max(Wpts[1, :]) + size * 1.25
    bbox = np.array([[min_x, max_x, max_x, min_x], [min_y, min_y, max_y, max_y]])

    # Find the homography matrix
    H, _ = dlt_homography(bbox, bpoly)
    
    img_pts = np.empty((Wpts.shape[0], Wpts.shape[1]))
    # Compute pts in image frame using H
    for i in range(Wpts.shape[1]):
        # Apply the transformation using matrix multiplication
        pt = np.matmul(H, Wpts[:, i] + np.array([0, 0, 1]))
        # Normalize the coordinates and store the point in the img_pts array
        img_pts[:, i] = pt / pt[2]
    # Round to integers
    img_pts = np.round(img_pts[:-1]).astype(int).T
        
    # Size of the square patches around each point
    patch_s = 17
    for i in range(Wpts.shape[1]):
        # Calculate the coordinates of the patch around the current point
        minx = img_pts[i, 0] - patch_s
        maxx = img_pts[i, 0] + patch_s
        miny = img_pts[i, 1] - patch_s
        maxy = img_pts[i, 1] + patch_s
        # Extract the patch from the image
        patch = I[miny:maxy, minx:maxx]

        # Find a saddle point and map the saddle point  back to image coordinates
        pt = saddle_point(patch) + np.array([[minx], [miny]])
        Ipts[:, i] = pt.reshape(2,)

    #------------------
    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts